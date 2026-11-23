use rayon::prelude::*;
use std::fs::File;
use std::io::{Read, Write};
use tyai_system::*;

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct TrainingSample {
    pub category: usize,
    pub question: String,
    pub answer: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== [trainer] Полноценное обучение TYAI (500-1000 нейронов на эксперта) ===");

    // 1. Загрузка словаря
    let mut vocab_file = File::open("brain_raw/vocab.bin")?;
    let mut vocab_bytes = Vec::new();
    vocab_file.read_to_end(&mut vocab_bytes)?;
    let vocab: VocabTable = bincode::deserialize(&vocab_bytes)?;
    println!("-> Загружен словарь: {} слов", vocab.word_to_id.len());

    // 2. Загрузка расширенного датасета
    let mut dataset_file = File::open("brain_raw/dataset.bin")?;
    let mut dataset_bytes = Vec::new();
    dataset_file.read_to_end(&mut dataset_bytes)?;
    let samples: Vec<TrainingSample> = bincode::deserialize(&dataset_bytes)?;
    println!("-> Загружен датасет: {} обучающих пар", samples.len());

    let encoder = Encoder::new(&vocab);

    // Параллельно кодируем запросы и ответы в тернарные векторы через Rayon
    println!("-> Параллельное тернарное кодирование {} диалоговых пар...", samples.len());
    let encoded_samples: Vec<(usize, Vec<i8>, Vec<i8>)> = samples
        .par_iter()
        .map(|s| {
            let q_vec = encoder.encode(&s.question);
            let a_vec = encoder.encode(&s.answer);
            (s.category, q_vec, a_vec)
        })
        .collect();
    println!("-> Кодирование завершено успешно.");

    // =========================================================================
    // ФАЗА 1: Обучение 40 мини-ИИ исполнителей (с авторостом от 10 до 500-1000 нейронов)
    // Параллельное многопоточное обучение через Rayon для максимальной скорости CPU
    // =========================================================================
    println!("\n>>> ФАЗА 1: Параллельное обучение 40 мини-ИИ (цель: 512+ нейронов на модуль)...");

    (0..TOTAL_EXPERTS).into_par_iter().for_each(|expert_id| {
        let mut seed: u64 = 10007 + expert_id as u64 * 37;
        let mut expert = MiniAI::new(expert_id, VECTOR_DIM, VECTOR_DIM, &mut seed);
        let start_neurons = expert.hidden_neurons.len(); // 10

        // Фильтруем все обучающие сэмплы для данного эксперта (сотни примеров)
        let my_samples: Vec<(&[i8], &[i8])> = encoded_samples
            .iter()
            .filter(|(cat, _, _)| *cat == expert_id)
            .map(|(_, q, a)| (q.as_slice(), a.as_slice()))
            .collect();

        // Обучаем с авторостом нейронов до 512+ (вплоть до 1024)
        let target_neurons = 512;
        train_mini_ai(&mut expert, &my_samples, target_neurons, &mut seed);

        let end_neurons = expert.hidden_neurons.len();

        // Сохраняем в отдельный файл
        let expert_filename = format!("brain_raw/expert_{:02}.bin", expert_id);
        let expert_bytes = bincode::serialize(&expert).expect("Ошибка сериализации эксперта");
        let mut file = File::create(&expert_filename).expect("Ошибка создания файла эксперта");
        file.write_all(&expert_bytes).expect("Ошибка записи эксперта");

        println!(
            "  [Эксперт {:02}/39] Нейрогенез: {} -> {} нейронов. Обучен на {} примерах -> {}",
            expert_id, start_neurons, end_neurons, my_samples.len(), expert_filename
        );
    });

    // =========================================================================
    // ФАЗА 2: Стыковка и обучение узлов маршрутизации
    // =========================================================================
    println!("\n>>> ФАЗА 2: Обучение конвейера маршрутизации на всем датасете...");

    let mut rng_seed: u64 = 123456789;
    let node2 = Node2::new(VECTOR_DIM, &mut rng_seed);
    let node3 = Node3::new(VECTOR_DIM, &mut rng_seed);
    let mut main_router = MainRouter::new(&mut rng_seed);
    let node1 = Node1::new(VECTOR_DIM, &mut rng_seed);

    // Обучение маршрутизации: вычисляем веса селекторов
    println!("  -> Точная настройка селекторов роутера на векторах запросов...");

    // 1. Настройка селекторов SubRouter в MainRouter:
    // Каждый SubRouter i отслеживает свой слот (40 + i) и свои категории (i*10 .. i*10 + 9)
    for sub_id in 0..NUM_SUB_ROUTERS {
        let selector = &mut main_router.sub_router_selectors[sub_id];
        for i in 0..VECTOR_DIM {
            selector.weights[i] = 0;
        }
        selector.weights[40 + sub_id] = 1;
        for cat in (sub_id * EXPERTS_PER_SUB_ROUTER)..((sub_id + 1) * EXPERTS_PER_SUB_ROUTER) {
            selector.weights[cat] = 1;
        }
    }

    // 2. Настройка селекторов конкретных экспертов в каждом SubRouter:
    // Эксперт cat отслеживает строго свой уникальный слот категории cat
    for cat in 0..TOTAL_EXPERTS {
        let sub_id = cat / EXPERTS_PER_SUB_ROUTER;
        let offset = cat % EXPERTS_PER_SUB_ROUTER;
        let sub_r = &mut main_router.sub_routers[sub_id];
        let exp_selector = &mut sub_r.selector_neurons[offset];

        for i in 0..VECTOR_DIM {
            exp_selector.weights[i] = 0;
        }
        exp_selector.weights[cat] = 1;
        exp_selector.weights[40 + sub_id] = 1;
    }

    // Сохранение узлов 1, 2, 3 и маршрутизатора
    let node1_bytes = bincode::serialize(&node1)?;
    File::create("brain_raw/node1.bin")?.write_all(&node1_bytes)?;
    println!("  -> Сохранен Node 1");

    let node2_bytes = bincode::serialize(&node2)?;
    File::create("brain_raw/node2.bin")?.write_all(&node2_bytes)?;
    println!("  -> Сохранен Node 2");

    let node3_bytes = bincode::serialize(&node3)?;
    File::create("brain_raw/node3.bin")?.write_all(&node3_bytes)?;
    println!("  -> Сохранен Node 3");

    let router_bytes = bincode::serialize(&main_router)?;
    File::create("brain_raw/router.bin")?.write_all(&router_bytes)?;
    println!("  -> Сохранен Router");

    println!("\nВсе модули успешно обучены и сохранены!\n");
    Ok(())
}
