use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use tyai_system::{VocabTable, TOTAL_EXPERTS};

/// Запись обучающей пары: вопрос, ответ и индекс эксперта (0..39)
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct TrainingSample {
    pub category: usize,
    pub question: String,
    pub answer: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== [gen_dataset] Импорт words.txt и dialogs.txt для обучения TYAI ===");

    let mut vocab = VocabTable::new();

    // 1. СНАЧАЛА регистрируем базовые якорные темы и расширенный набор разговорных пар,
    // чтобы слова тем получили приоритетные категории (слоты 0..40)!
    let anchors: Vec<(usize, &str, &str, Vec<&str>)> = vec![
        // Системные и диагностические
        (0, "система в норме", "статус системы", vec!["проверить систему", "состояние системы", "проверка ядра", "как система"]),
        (1, "память оптимизирована", "память", vec!["буфер памяти", "состояние памяти", "кэш память", "оперативная память"]),
        (2, "cpu работает штатно", "процессор", vec!["нагрузка cpu", "ядра процессора", "тактовая частота", "загрузка процессора"]),
        (3, "накопитель в порядке", "диск", vec!["дисковое пространство", "накопитель ssd", "проверка диска", "место на диске"]),
        (4, "сеть стабильна", "интернет", vec!["проверка сети", "пинг", "соединение", "сетевой статус"]),

        // Математика и базовые операции
        (10, "сумма вычислена", "сложение", vec!["плюс", "прибавить", "суммировать", "посчитай сумму"]),
        (11, "разность готова", "вычитание", vec!["минус", "отнять", "вычесть", "посчитай разность"]),

        // Разговорный блок: Приветствия
        (20, "приветствую тебя! рад общению", "привет", vec![
            "здравствуй", "здравствуйте", "добрый день", "доброе утро", "добрый вечер", "салют", "хай", "хеллоу", "приветик", "приветствую", "ку"
        ]),

        // Разговорный блок: Состояние / настроение
        (21, "все отлично готов к работе и общению", "как дела", vec![
            "как поживаешь", "как настроение", "все ли хорошо", "как ты", "как жизнь", "как успехи", "как делишки", "че как", "как оно"
        ]),

        // Разговорный блок: Самоидентификация
        (22, "я модульный саморастущий ии tyai", "кто ты", vec![
            "что ты такое", "представься", "расскажи о себе", "как тебя зовут", "ты кто", "ты человек или робот", "ты бот", "кто тебя создал"
        ]),

        // Разговорный блок: Благодарность
        (23, "всегда пожалуйста рад помочь", "спасибо", vec![
            "благодарю", "большое спасибо", "рад помочь", "от души", "спасибо большое", "пасиб", "сенкс"
        ]),

        // Разговорный блок: Прощание
        (24, "до встречи хорошего дня", "пока", vec![
            "до свидания", "прощай", "до скорого", "увидимся", "спокойной ночи", "бывай", "до связи"
        ]),

        // Разговорный блок: Возможности
        (25, "я умею отвечать на вопросы вести диалог и расти", "что ты умеешь", vec![
            "каковы твои возможности", "чем можешь помочь", "что умеешь делать", "на что способен", "помощь", "справка", "команды"
        ]),

        // Разговорный блок: Текущее действие
        (26, "обрабатываю сигналы и общаюсь с тобой", "что делаешь", vec![
            "чем занят", "что поделываешь", "чем занимаешься", "что нового"
        ]),

        // Разговорный блок: Время / дата
        (27, "время двигаться вперед и развиваться", "сколько времени", vec![
            "который час", "какое сейчас время", "скажи время", "какой сегодня день"
        ]),

        // Разговорный блок: Бытовые вопросы и комплименты
        (28, "спасибо приятно слышать стараюсь для тебя", "ты молодец", vec![
            "ты умный", "отличная работа", "круто", "ты супер", "молодец", "хороший ответ"
        ]),
        (29, "я здесь и внимательно слушаю тебя", "ты тут", vec![
            "ты здесь", "ты на связи", "ответь", "ты работаешь", "ау"
        ]),

        // Архитектура и движок TYAI
        (30, "написан на rust без фреймворков", "язык программирования", vec!["rust", "почему rust", "на чем написан", "исходный код"]),
        (31, "веса только минус один ноль и один", "тернарные веса", vec!["тернарная логика", "веса сети", "тернарные числа"]),
        (32, "автодобавление нейронов при стагнации", "саморост нейронов", vec!["нейрогенез", "рост нейронов", "добавление нейронов"]),
    ];

    let mut samples: Vec<TrainingSample> = Vec::new();

    // Регистрируем якорные и разговорные реплики с ПОВЫШЕННЫМ ВЕСОМ (множественные копии)
    for (cat, canonical_ans, trigger, variations) in &anchors {
        let cat = *cat;
        vocab.register_category_response(cat, canonical_ans);
        for w in trigger.split_whitespace() {
            vocab.add_word_with_category(w, Some(cat));
        }
        for v in variations {
            for w in v.split_whitespace() {
                vocab.add_word_with_category(w, Some(cat));
            }
        }

        // Добавляем якорную пару трижды для усиления веса в обучении
        for _ in 0..5 {
            samples.push(TrainingSample {
                category: cat,
                question: trigger.to_string(),
                answer: canonical_ans.to_string(),
            });
        }

        for v in variations {
            for _ in 0..3 {
                samples.push(TrainingSample {
                    category: cat,
                    question: v.to_string(),
                    answer: canonical_ans.to_string(),
                });
            }
        }
    }

    // 2. Загрузка 10 000 слов из words.txt в словарь (без перезаписи категорий)
    println!("1. Чтение словаря words.txt...");
    if let Ok(file) = File::open("words.txt") {
        let reader = BufReader::new(file);
        let mut count = 0;
        for line in reader.lines() {
            if let Ok(word) = line {
                let clean = word.trim().to_lowercase();
                if !clean.is_empty() {
                    vocab.add_word(&clean);
                    count += 1;
                }
            }
        }
        println!("  -> Загружено {} слов из words.txt", count);
    } else {
        eprintln!("Внимание: файл words.txt не найден!");
    }

    // Функция определения категории эксперта для реплики
    let classify_query = |text: &str, current_vocab: &VocabTable| -> usize {
        let clean = text.to_lowercase();
        // Приоритетные ключевые слова (с учетом частых опечаток и разговорных форм)
        if clean.contains("привет") || clean.contains("превет") || clean.contains("прив") || clean.contains("здравствуй") 
            || clean.contains("салют") || clean.contains("добрый день") || clean.contains("доброе утро") || clean.contains("вечер") 
            || clean.contains("хай") || clean.contains("хеллоу") {
            return 20;
        }
        if clean.contains("как дела") || clean.contains("как делаешь") || clean.contains("как поживаешь") 
            || clean.contains("как жизнь") || clean.contains("как настроение") || clean.contains("как ты") || clean.contains("че как") {
            return 21;
        }
        if clean.contains("кто ты") || clean.contains("что ты") || clean.contains("ты кто") 
            || clean.contains("представься") || clean.contains("твое имя") || clean.contains("как тебя зовут") 
            || clean.contains("расскажи о себе") || clean.contains("ты человек") || clean.contains("ты бот") {
            return 22;
        }
        if clean.contains("спасибо") || clean.contains("спасиб") || clean.contains("благодарю") || clean.contains("сенкс") {
            return 23;
        }
        if clean.contains("пока") || clean.contains("до свидания") || clean.contains("прощай") || clean.contains("до встречи") || clean.contains("увидимся") {
            return 24;
        }
        if clean.contains("что ты умеешь") || clean.contains("что умеешь") || clean.contains("помощь") || clean.contains("справка") || clean.contains("команды") || clean.contains("возможности") {
            return 25;
        }
        if clean.contains("что делаешь") || clean.contains("чем занят") || clean.contains("чем занимаешься") || clean.contains("что поделываешь") {
            return 26;
        }
        if clean.contains("времени") || clean.contains("который час") || clean.contains("время") || clean.contains("дата") || clean.contains("число") {
            return 27;
        }
        if clean.contains("молодец") || clean.contains("умный") || clean.contains("умница") || clean.contains("супер") || clean.contains("круто") {
            return 28;
        }
        if clean.contains("ты тут") || clean.contains("ты здесь") || clean.contains("на связи") || clean.contains("ответь") {
            return 29;
        }
        if clean.contains("статус") || clean.contains("система") || clean.contains("хост") {
            return 0;
        }
        if clean.contains("памят") || clean.contains("буфер") || clean.contains("кэш") || clean.contains("ram") {
            return 1;
        }
        if clean.contains("процессор") || clean.contains("cpu") || clean.contains("ядра") {
            return 2;
        }
        if clean.contains("диск") || clean.contains("накопител") || clean.contains("ssd") {
            return 3;
        }
        if clean.contains("сеть") || clean.contains("интернет") || clean.contains("пинг") {
            return 4;
        }
        if clean.contains("плюс") || clean.contains("слож") || clean.contains("прибав") {
            return 10;
        }
        if clean.contains("минус") || clean.contains("вычит") || clean.contains("отнят") {
            return 11;
        }
        if clean.contains("rust") || clean.contains("язык") {
            return 30;
        }
        if clean.contains("тернарн") || clean.contains("вес") {
            return 31;
        }
        if clean.contains("нейрогенез") || clean.contains("рост") {
            return 32;
        }

        // Если есть совпадение со словами словаря, у которых уже есть категория
        for w in clean.split_whitespace() {
            if let Some(&id) = current_vocab.word_to_id.get(w) {
                if let Some(v) = current_vocab.word_vectors.get(&id) {
                    for cat in 0..40 {
                        if v[cat] > 0 {
                            return cat;
                        }
                    }
                }
            }
        }

        // Хэш по умолчанию распределяет остальные реплики по экспертам диалогов
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in clean.as_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        (h % (TOTAL_EXPERTS as u64)) as usize
    };

    // 3. Загрузка и распределение диалогов из dialogs.txt
    println!("2. Чтение диалогов dialogs.txt (до 50 000 строк)...");
    if let Ok(file) = File::open("dialogs.txt") {
        let reader = BufReader::new(file);
        let mut total_dialogs = 0;

        for line in reader.lines() {
            if let Ok(l) = line {
                if let Some((q_raw, a_raw)) = l.split_once('|') {
                    let q = q_raw.trim().trim_start_matches('-').trim().to_string();
                    let a = a_raw.trim().trim_start_matches('-').trim().to_string();

                    if q.is_empty() || a.is_empty() || q.len() > 120 || a.len() > 150 {
                        continue;
                    }

                    // Добавляем слова вопроса и ответа в словарь
                    for w in q.split_whitespace() {
                        vocab.add_word(w);
                    }
                    for w in a.split_whitespace() {
                        vocab.add_word(w);
                    }

                    let category = classify_query(&q, &vocab);

                    samples.push(TrainingSample {
                        category,
                        question: q,
                        answer: a,
                    });

                    total_dialogs += 1;
                    if total_dialogs >= 40000 {
                        break;
                    }
                }
            }
        }
        println!("  -> Успешно импортировано {} диалоговых пар", total_dialogs);
    } else {
        eprintln!("Внимание: файл dialogs.txt не найден!");
    }

    // Регистрируем диалоговые ответы в VocabTable для каждого эксперта
    println!("3. Регистрация диалоговых ответов в VocabTable...");
    let precomputed_dialogs: Vec<(usize, Vec<i8>, String)> = {
        let encoder = tyai_system::Encoder::new(&vocab);
        samples
            .iter()
            .map(|s| (s.category, encoder.encode(&s.question), s.answer.clone()))
            .collect()
    };

    for (cat, q_vec, ans_text) in precomputed_dialogs {
        vocab.register_expert_dialog_response(cat, q_vec, ans_text);
    }

    // 4. Сохранение артефактов
    println!("4. Сохранение словаря и датасета в brain_raw/...");
    let vocab_bytes = bincode::serialize(&vocab)?;
    File::create("brain_raw/vocab.bin")?.write_all(&vocab_bytes)?;
    println!("  -> Сохранен brain_raw/vocab.bin (всего слов: {})", vocab.word_to_id.len());

    let dataset_bytes = bincode::serialize(&samples)?;
    File::create("brain_raw/dataset.bin")?.write_all(&dataset_bytes)?;
    println!("  -> Сохранен brain_raw/dataset.bin (всего сэмплов: {})", samples.len());

    println!("Генерация датасета успешно завершена!\n");
    Ok(())
}
