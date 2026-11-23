use memmap2::MmapOptions;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::time::Instant;
use tyai_system::*;

struct TyaiEngine {
    _file: File,
    mmap: memmap2::Mmap,
    header: TyaiHeader,
    index: HashMap<u32, TyaiIndexEntry>,
    vocab: VocabTable,
    node1: Node1,
    node2: Node2,
    node3: Node3,
    main_router: MainRouter,
    dispatcher: Dispatcher,
    // Счетчики сессии
    cache_hits: usize,
    nn_calls: usize,
}

impl TyaiEngine {
    pub fn load(model_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(model_path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // 1. Читаем заголовок Header
        let header_size = std::mem::size_of::<TyaiHeader>();
        let header: TyaiHeader = bincode::deserialize(&mmap[0..header_size])?;

        if &header.magic != TYAI_MAGIC {
            return Err("Неверный формат файла: отсутствует сигнатура TYAI".into());
        }

        // 2. Читаем Index Table
        let mut index = HashMap::new();
        let entry_size = bincode::serialized_size(&TyaiIndexEntry { module_id: 0, offset: 0, size: 0 })? as usize;
        let mut offset = header.index_offset as usize;

        for _ in 0..header.index_count {
            let entry: TyaiIndexEntry = bincode::deserialize(&mmap[offset..offset + entry_size])?;
            index.insert(entry.module_id, entry);
            offset += entry_size;
        }

        // 3. Загружаем постоянные легковесные модули (Vocab, Node 1, 2, 3, MainRouter)
        // Эксперты (0..39) в память НЕ загружаются! Они читаются по требованию из mmap.
        let vocab_entry = index.get(&MODULE_ID_VOCAB).ok_or("Словарь не найден в .tyai")?;
        let vocab: VocabTable = bincode::deserialize(
            &mmap[vocab_entry.offset as usize..(vocab_entry.offset + vocab_entry.size) as usize],
        )?;

        let n1_entry = index.get(&MODULE_ID_NODE1).ok_or("Node 1 не найден")?;
        let node1: Node1 = bincode::deserialize(
            &mmap[n1_entry.offset as usize..(n1_entry.offset + n1_entry.size) as usize],
        )?;

        let n2_entry = index.get(&MODULE_ID_NODE2).ok_or("Node 2 не найден")?;
        let node2: Node2 = bincode::deserialize(
            &mmap[n2_entry.offset as usize..(n2_entry.offset + n2_entry.size) as usize],
        )?;

        let n3_entry = index.get(&MODULE_ID_NODE3).ok_or("Node 3 не найден")?;
        let node3: Node3 = bincode::deserialize(
            &mmap[n3_entry.offset as usize..(n3_entry.offset + n3_entry.size) as usize],
        )?;

        let r_entry = index.get(&MODULE_ID_ROUTER).ok_or("Router не найден")?;
        let main_router: MainRouter = bincode::deserialize(
            &mmap[r_entry.offset as usize..(r_entry.offset + r_entry.size) as usize],
        )?;

        Ok(Self {
            _file: file,
            mmap,
            header,
            index,
            vocab,
            node1,
            node2,
            node3,
            main_router,
            dispatcher: Dispatcher::new(),
            cache_hits: 0,
            nn_calls: 0,
        })
    }

    /// Ленивая загрузка одного конкретного эксперта через mmap без загрузки остальных 39!
    fn load_expert(&self, expert_id: usize) -> Result<MiniAI, Box<dyn std::error::Error>> {
        let entry = self
            .index
            .get(&(expert_id as u32))
            .ok_or_else(|| format!("Эксперт {} не найден в индексе .tyai", expert_id))?;
        let start = entry.offset as usize;
        let end = (entry.offset + entry.size) as usize;
        let expert: MiniAI = bincode::deserialize(&self.mmap[start..end])?;
        Ok(expert)
    }

    /// Подсчет суммарного числа нейронов во всех 40 экспертах в контейнере
    pub fn get_total_expert_neurons(&self) -> (usize, usize) {
        let mut total_neurons = 0;
        let mut loaded_count = 0;
        for exp_id in 0..TOTAL_EXPERTS {
            if let Ok(exp) = self.load_expert(exp_id) {
                total_neurons += exp.hidden_neurons.len();
                loaded_count += 1;
            }
        }
        (total_neurons, loaded_count)
    }

    /// Очистка кэш-буфера диспетчера
    pub fn clear_cache(&mut self) -> usize {
        let count = self.dispatcher.cache.entries.len();
        self.dispatcher.cache.clear();
        count
    }

    /// Обучение на лету / добавление пары в кэш и в dialogs.txt
    pub fn learn_dialog(&mut self, question: &str, answer: &str) -> Result<(), Box<dyn std::error::Error>> {
        let encoder = Encoder::new(&self.vocab);
        let q_code = encoder.encode(question);
        let a_code = encoder.encode(answer);

        // 1. Сохраняем в кэш-буфер диспетчера для немедленного использования
        self.dispatcher.cache.insert(
            question.to_string(),
            q_code,
            answer.to_string(),
            a_code,
        );

        // 2. Дописываем пару в конец dialogs.txt для будущих переобучений
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("dialogs.txt")?;
        writeln!(file, "{} | {}", question, answer)?;

        Ok(())
    }

    /// Полный цикл прохождения сигнала от User до вывода
    pub fn process_query(&mut self, query: &str) -> (String, u128, String) {
        let start_time = Instant::now();

        // ---------------------------------------------------------------------
        // ШАГ 1: User <-> Dispatcher <-> Cache Buffer
        // ---------------------------------------------------------------------
        let encoder = Encoder::new(&self.vocab);
        let query_code = encoder.encode(query);

        if let Some(cached) = self.dispatcher.cache.lookup(query, Some(&query_code)) {
            self.cache_hits += 1;
            let elapsed_us = start_time.elapsed().as_micros();
            let log = "[КЭШ-БУФЕР] Мгновенное попадание в память (нейросети не запускались)".to_string();
            return (cached.response.clone(), elapsed_us, log);
        }

        self.nn_calls += 1;

        // ---------------------------------------------------------------------
        // ШАГ 2: Encoder <-> Vocab Table -> Node 2 & Node 3
        // ---------------------------------------------------------------------
        // Параллельное извлечение смысла/действия (Node 2) и объектов/контекста (Node 3)
        let sense_vec = self.node2.forward(&query_code);
        let context_vec = self.node3.forward(&query_code);

        // Объединение признаков для Главного распределителя
        let mut router_input = vec![0i8; VECTOR_DIM];
        for i in 0..VECTOR_DIM {
            router_input[i] = clamp_ternary(sense_vec[i] + context_vec[i]);
        }

        // ---------------------------------------------------------------------
        // ШАГ 3: Иерархическое дерево маршрутизации
        // Main Router -> выбор 1 или 2 Sub-Routers -> выбор 1-2 Mini-AI
        // ---------------------------------------------------------------------
        let active_sub_routers = self.main_router.route(&router_input);
        let mut activated_experts = Vec::new();

        for &sub_id in &active_sub_routers {
            let sub_router = &self.main_router.sub_routers[sub_id];
            let expert_id = sub_router.select_expert(&router_input);
            activated_experts.push((sub_id, expert_id));
        }

        // Вычисление ответов выбранными мини-ИИ и возврат по двусторонним связям
        let mut expert_responses: Vec<Vec<i8>> = Vec::new();
        let mut log_modules = Vec::new();

        for &(sub_id, exp_id) in &activated_experts {
            // Загружаем ИСКЛЮЧИТЕЛЬНО этот мини-ИИ через mmap
            if let Ok(expert) = self.load_expert(exp_id) {
                let code_out = expert.forward(&query_code);
                expert_responses.push(code_out);
                log_modules.push(format!(
                    "Sub-Router #{} -> Mini-AI #{} (активных нейронов: {})",
                    sub_id,
                    exp_id,
                    expert.hidden_neurons.len()
                ));
            }
        }

        // ---------------------------------------------------------------------
        // ШАГ 4: Сборка и декодирование
        // Main Router -> Node 1 -> Decoder -> Buffer -> Dispatcher -> User
        // ---------------------------------------------------------------------
        // Агрегируем отклики экспертов
        let mut combined_code = vec![0i32; VECTOR_DIM];
        for resp in &expert_responses {
            for i in 0..VECTOR_DIM.min(resp.len()) {
                combined_code[i] += resp[i] as i32;
            }
        }
        let aggregated_ternary: Vec<i8> = combined_code
            .into_iter()
            .map(|val| threshold_activation(val, 0))
            .collect();

        // Node 1: финальное выравнивание кодового вектора
        let aligned_code = self.node1.forward(&aggregated_ternary);

        // Decoder: сверка с Vocab Table и перевод в текст
        let primary_expert = activated_experts.first().map(|&(_, exp_id)| exp_id);
        let decoder = Decoder::new(&self.vocab);
        let answer_text = decoder.decode_with_category(&aligned_code, &query_code, primary_expert);

        // Запись в кэш-буфер диспетчера для мгновенных повторных ответов
        self.dispatcher.cache.insert(
            query.to_string(),
            query_code,
            answer_text.clone(),
            aligned_code,
        );

        let elapsed_us = start_time.elapsed().as_micros();
        let execution_log = format!("[МАРШРУТИЗАЦИЯ] {}", log_modules.join(" | "));

        (answer_text, elapsed_us, execution_log)
    }
}

fn print_welcome_banner(model_path: &str, engine: &TyaiEngine) {
    println!("============================================================");
    println!("     TYAI v1.0: Терминальный клиент запуска саморастущего ИИ    ");
    println!("============================================================");
    println!("Загрузка контейнера: {}", model_path);
    println!(
        "Модель успешно смонтирована (mmap). Версия: v{}, модулей: {}",
        engine.header.version, engine.header.total_modules
    );
    println!("Тернарное ядро CPU готово. Память ОЗУ свободна.");
    println!("------------------------------------------------------------");
    println!("Доступные служебные команды:");
    println!("  /stats                    — статистика модели и сессии");
    println!("  /clear                    — очистка кэш-буфера диспетчера");
    println!("  /learn <вопрос> | <ответ> — обучить модель ответу прямо в чате");
    println!("  exit / выход              — завершить сессию");
    println!("------------------------------------------------------------");
    println!("Введите текстовый запрос для общения с ИИ:");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = "model.tyai".to_string();

    for i in 1..args.len() {
        if args[i] == "--model" && i + 1 < args.len() {
            model_path = args[i + 1].clone();
        }
    }

    let mut engine = match TyaiEngine::load(&model_path) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Ошибка загрузки модели {}: {}", model_path, e);
            eprintln!("Подсказка: сначала выполните gen_dataset -> trainer -> packer");
            return Ok(());
        }
    };

    print_welcome_banner(&model_path, &engine);

    let stdin = io::stdin();
    let mut input = String::new();

    loop {
        print!("\n> ");
        io::stdout().flush()?;
        input.clear();

        if stdin.read_line(&mut input)? == 0 {
            break; // EOF
        }

        let query = input.trim();
        if query.is_empty() {
            continue;
        }

        if query == "exit" || query == "quit" || query == "выход" {
            println!("Завершение сессии TYAI. До встречи!");
            break;
        }

        // Команда /stats
        if query == "/stats" {
            let (total_neurons, loaded_experts) = engine.get_total_expert_neurons();
            println!("\n=== [СВОДКА TYAI МОДЕЛИ] ===");
            println!("  Формат файла         : {}", String::from_utf8_lossy(&engine.header.magic));
            println!("  Версия формата       : v{}", engine.header.version);
            println!("  Всего модулей        : {}", engine.header.total_modules);
            println!("  Размер словаря       : {} слов / токенов", engine.vocab.word_to_id.len());
            println!("  Экспертов просканировано: {} из {}", loaded_experts, TOTAL_EXPERTS);
            println!("  Суммарно нейронов в 40 мини-ИИ: {}", total_neurons);
            println!("  Записей в кэш-буфере : {}", engine.dispatcher.cache.entries.len());
            println!("  Попаданий в кэш      : {}", engine.cache_hits);
            println!("  Вызовов нейросетей   : {}", engine.nn_calls);
            let total_reqs = engine.cache_hits + engine.nn_calls;
            if total_reqs > 0 {
                let hit_rate = (engine.cache_hits as f64 / total_reqs as f64) * 100.0;
                println!("  Эффективность кэша   : {:.1}%", hit_rate);
            }
            continue;
        }

        // Команда /clear
        if query == "/clear" {
            let cleared = engine.clear_cache();
            println!("[СИСТЕМА] Оперативный кэш-буфер диспетчера очищен (удалено записей: {}).", cleared);
            continue;
        }

        // Команда /learn
        if query.starts_with("/learn") {
            let rest = query.trim_start_matches("/learn").trim();
            if let Some((q, a)) = rest.split_once('|') {
                let q_clean = q.trim();
                let a_clean = a.trim();
                if q_clean.is_empty() || a_clean.is_empty() {
                    println!("[ОШИБКА] Неверный формат. Используйте: /learn <вопрос> | <ответ>");
                } else {
                    match engine.learn_dialog(q_clean, a_clean) {
                        Ok(()) => {
                            println!("[ОБУЧЕНИЕ] Успешно! Фраза добавлена в кэш-буфер и сохранена в dialogs.txt:");
                            println!("  Вопрос: \"{}\"", q_clean);
                            println!("  Ответ : \"{}\"", a_clean);
                        }
                        Err(e) => {
                            println!("[ОШИБКА] Не удалось сохранить пару: {}", e);
                        }
                    }
                }
            } else {
                println!("[ОШИБКА] Неверный формат. Используйте: /learn <вопрос> | <ответ>");
            }
            continue;
        }

        let (answer, elapsed_us, route_log) = engine.process_query(query);

        println!("Ответ ИИ : {}", answer);
        println!("Время    : {} мкс ({:.3} мс)", elapsed_us, elapsed_us as f64 / 1000.0);
        println!("Лог сети : {}", route_log);
    }

    Ok(())
}
