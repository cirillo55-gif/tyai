//! # Модульная саморастущая нейросеть TYAI
//!
//! Архитектура конвейера:
//! User <-> Dispatcher (Cache Buffer)
//!   -> Encoder <-> Vocab Table
//!     -> Node 2 (смысл/действие) & Node 3 (объекты/контекст)
//!       -> Main Router
//!         -> Sub-Routers 1..4 (выбор 1..2 активных веток)
//!           -> Executors (40 мини-ИИ, по 10 на каждый Sub-Router: 2 группы по 5)
//!       <- возврат кодов по двусторонним связям
//!   -> Node 1 (агрегация и выравнивание)
//!   -> Decoder <-> Vocab Table
//!   -> Buffer -> Dispatcher -> User

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Размерность эмбеддингов/кодовых векторов в системе (40 слотов категорий + 8 саброутеров + 48 бит хэша)
pub const VECTOR_DIM: usize = 96;

/// Магические байты для формата .tyai
pub const TYAI_MAGIC: &[u8; 4] = b"TYAI";
/// Текущая версия формата .tyai
pub const TYAI_VERSION: u32 = 1;
/// Общее количество модулей/экспертов в системе
pub const TOTAL_EXPERTS: usize = 40;
/// Количество Sub-Router'ов
pub const NUM_SUB_ROUTERS: usize = 4;
/// Количество экспертов на один Sub-Router
pub const EXPERTS_PER_SUB_ROUTER: usize = 10;

// ---------------------------------------------------------------------------
// 1. Тернарные типы и математическое ядро (без f32/f64 и умножений)
// ---------------------------------------------------------------------------

/// Тернарный вес: может принимать значения только -1, 0, 1.
/// Хранится компактно как i8.
pub type TernaryWeight = i8;

/// Проверка валидности тернарного веса
#[inline(always)]
pub fn clamp_ternary(w: i8) -> TernaryWeight {
    if w > 0 {
        1
    } else if w < 0 {
        -1
    } else {
        0
    }
}

/// Пороговая функция активации:
/// Если сумма > threshold -> +1,
/// Если сумма < -threshold -> -1,
/// Иначе -> 0.
#[inline(always)]
pub fn threshold_activation(sum: i32, threshold: i32) -> i8 {
    if sum > threshold {
        1
    } else if sum < -threshold {
        -1
    } else {
        0
    }
}

/// Нейрон с тернарными входящими весами и порогом.
/// Все вычисления происходят исключительно через сложение и вычитание целых чисел.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neuron {
    /// Входящие веса связей (только -1, 0, 1)
    pub weights: Vec<TernaryWeight>,
    /// Порог активации (целое число)
    pub threshold: i32,
    /// Флаг активности/значимости для механизма обрезки (pruning)
    pub is_active: bool,
}

impl Neuron {
    /// Создание нового нейрона с заданным количеством входных связей
    pub fn new(input_dim: usize, threshold: i32) -> Self {
        Self {
            weights: vec![0; input_dim],
            threshold,
            is_active: true,
        }
    }

    /// Создание нейрона со случайными тернарными связями
    pub fn random(input_dim: usize, threshold: i32, seed: &mut u64) -> Self {
        let mut weights = Vec::with_capacity(input_dim);
        for _ in 0..input_dim {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = ((*seed >> 33) % 3) as i8 - 1; // -1, 0, 1
            weights.push(val);
        }
        Self {
            weights,
            threshold,
            is_active: true,
        }
    }

    /// Прямой проход нейрона: вычисление без умножений!
    /// Используются только операторы сложения и вычитания.
    #[inline(always)]
    pub fn forward(&self, inputs: &[i8]) -> i8 {
        if !self.is_active {
            return 0;
        }
        let mut sum: i32 = 0;
        let len = self.weights.len().min(inputs.len());
        for i in 0..len {
            match self.weights[i] {
                1 => sum += inputs[i] as i32,
                -1 => sum -= inputs[i] as i32,
                _ => {} // умножение на 0 пропускается
            }
        }
        threshold_activation(sum, self.threshold)
    }
}

// ---------------------------------------------------------------------------
// 2. Мини-ИИ (Executor) с конструктивным саморостом нейронов
// ---------------------------------------------------------------------------

/// Мини-ИИ нейронная сеть (Executor).
/// Стартует с 10 нейронов, умеет динамически выращивать новые нейроны
/// и производить обрезку (pruning) неактивных/зануленных нейронов.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiniAI {
    pub id: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    /// Скрытый слой нейронов, способный к саморосту (от 10 до max_neurons)
    pub hidden_neurons: Vec<Neuron>,
    /// Выходные нейроны, формирующие итоговый вектор
    pub output_neurons: Vec<Neuron>,
    /// Максимальное число нейронов в скрытом слое
    pub max_neurons: usize,
    /// Счётчик эпох без улучшений (для триггера нейрогенеза)
    pub stagnation_counter: usize,
}

impl MiniAI {
    /// Инициализация Мини-ИИ с 10 начальными скрытыми нейронами
    pub fn new(id: usize, input_dim: usize, output_dim: usize, seed: &mut u64) -> Self {
        let initial_neurons = 10;
        let mut hidden = Vec::with_capacity(initial_neurons);
        for _ in 0..initial_neurons {
            hidden.push(Neuron::random(input_dim, 0, seed));
        }

        let mut output = Vec::with_capacity(output_dim);
        for _ in 0..output_dim {
            output.push(Neuron::random(initial_neurons, 0, seed));
        }

        Self {
            id,
            input_dim,
            output_dim,
            hidden_neurons: hidden,
            output_neurons: output,
            max_neurons: 1024,
            stagnation_counter: 0,
        }
    }

    /// Прямой проход через Мини-ИИ (без f32 умножений)
    pub fn forward(&self, input: &[i8]) -> Vec<i8> {
        let mut hidden_acts = Vec::with_capacity(self.hidden_neurons.len());
        for neuron in &self.hidden_neurons {
            hidden_acts.push(neuron.forward(input));
        }

        let mut outputs = Vec::with_capacity(self.output_neurons.len());
        for neuron in &self.output_neurons {
            outputs.push(neuron.forward(&hidden_acts));
        }
        outputs
    }

    /// Авторост нейронов (Конструктивный нейрогенез)
    /// Добавляет `count` новых скрытых нейронов со случайными связями, если лимит не исчерпан
    pub fn spawn_neurons(&mut self, count: usize, seed: &mut u64) -> usize {
        let mut added = 0;
        for _ in 0..count {
            if self.hidden_neurons.len() >= self.max_neurons {
                break;
            }
            let new_neuron = Neuron::random(self.input_dim, 0, seed);
            self.hidden_neurons.push(new_neuron);

            for out_n in &mut self.output_neurons {
                *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let w = ((*seed >> 33) % 3) as i8 - 1;
                out_n.weights.push(w);
            }
            added += 1;
        }

        self.stagnation_counter = 0;
        added
    }

    pub fn spawn_neuron(&mut self, seed: &mut u64) -> bool {
        self.spawn_neurons(1, seed) > 0
    }

    /// Механизм обрезки (Pruning):
    /// Удаляет нейроны, чьи веса занулены или не влияют на выходные нейроны
    pub fn prune(&mut self) -> usize {
        let min_required = 500.min(self.hidden_neurons.len());
        if self.hidden_neurons.len() <= min_required {
            return 0; // Сохраняем минимум
        }

        let mut keep_indices = Vec::new();
        for (i, h_neuron) in self.hidden_neurons.iter().enumerate() {
            let mut connected_to_output = false;
            for out_n in &self.output_neurons {
                if i < out_n.weights.len() && out_n.weights[i] != 0 {
                    connected_to_output = true;
                    break;
                }
            }
            let has_incoming = h_neuron.weights.iter().any(|&w| w != 0);

            if connected_to_output && has_incoming {
                keep_indices.push(i);
            }
        }

        if keep_indices.len() < min_required {
            return 0;
        }

        let removed = self.hidden_neurons.len() - keep_indices.len();
        let new_hidden: Vec<Neuron> = keep_indices
            .iter()
            .map(|&idx| self.hidden_neurons[idx].clone())
            .collect();

        for out_n in &mut self.output_neurons {
            let new_weights: Vec<i8> = keep_indices
                .iter()
                .map(|&idx| if idx < out_n.weights.len() { out_n.weights[idx] } else { 0 })
                .collect();
            out_n.weights = new_weights;
        }

        self.hidden_neurons = new_hidden;
        removed
    }
}

// ---------------------------------------------------------------------------
// 3. Таблица словаря и Кодировщик / Декодировщик (Vocab Table, Encoder, Decoder)
// ---------------------------------------------------------------------------

/// Таблица кодировки и раскодировки (Vocab Table)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabTable {
    pub word_to_id: HashMap<String, u32>,
    pub id_to_word: HashMap<u32, String>,
    pub word_vectors: HashMap<u32, Vec<i8>>,
    /// Соответствие категории (0..39) шаблону ответа для декодера
    pub category_templates: HashMap<usize, String>,
    /// База диалоговых пар (вектор вопроса -> текст ответа) по каждому эксперту
    pub expert_dialog_responses: HashMap<usize, Vec<(Vec<i8>, String)>>,
}

impl VocabTable {
    pub fn new() -> Self {
        Self {
            word_to_id: HashMap::new(),
            id_to_word: HashMap::new(),
            word_vectors: HashMap::new(),
            category_templates: HashMap::new(),
            expert_dialog_responses: HashMap::new(),
        }
    }

    /// Регистрация диалоговой пары (вопрос -> ответ) для эксперта
    pub fn register_expert_dialog_response(&mut self, category: usize, query_vec: Vec<i8>, answer_text: String) {
        let list = self.expert_dialog_responses.entry(category).or_insert_with(Vec::new);
        if list.len() < 1000 {
            list.push((query_vec, answer_text));
        }
    }

    /// Регистрация типового ответа для категории эксперта
    pub fn register_category_response(&mut self, category: usize, response: &str) {
        self.category_templates.insert(category, response.to_string());
    }

    /// Добавление слова в словарь с привязкой к семантической теме (категории эксперта)
    pub fn add_word_with_category(&mut self, word: &str, category: Option<usize>) -> u32 {
        let clean = word.to_lowercase().trim().to_string();
        let id = if let Some(&existing_id) = self.word_to_id.get(&clean) {
            existing_id
        } else {
            let new_id = self.word_to_id.len() as u32;
            self.word_to_id.insert(clean.clone(), new_id);
            self.id_to_word.insert(new_id, clean.clone());
            let mut vec = vec![0i8; VECTOR_DIM];

            // Индивидуальный тернарный хэш слова в диапазоне [48..96]
            let mut h: u64 = 0xcbf29ce484222325;
            for &b in clean.as_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            for i in 48..VECTOR_DIM {
                h = h.wrapping_mul(6364136223846793005).wrapping_add(1);
                let val = ((h >> 33) % 3) as i8 - 1;
                vec[i] = val;
            }
            self.word_vectors.insert(new_id, vec);
            new_id
        };

        // Если передана категория, активируем ортогональный слот категории и ее саброутера
        if let Some(cat) = category {
            if cat < 40 {
                if let Some(vec) = self.word_vectors.get_mut(&id) {
                    vec[cat] = 1;
                    let sub_router_id = cat / 10;
                    vec[40 + sub_router_id] = 1;
                }
            }
        }

        id
    }

    /// Добавление слова в словарь с генерацией детерминированного тернарного вектора
    pub fn add_word(&mut self, word: &str) -> u32 {
        self.add_word_with_category(word, None)
    }

    /// Поиск ближайшего слова к заданному тернарному вектору
    pub fn find_closest_word(&self, target_vec: &[i8]) -> Option<String> {
        let mut best_score = i32::MIN;
        let mut best_word = None;

        for (id, vec) in &self.word_vectors {
            let mut score = 0i32;
            for i in 0..VECTOR_DIM.min(vec.len()).min(target_vec.len()) {
                if vec[i] == target_vec[i] && vec[i] != 0 {
                    score += 2;
                } else if vec[i] == -target_vec[i] && vec[i] != 0 {
                    score -= 2;
                }
            }
            if score > best_score {
                best_score = score;
                best_word = self.id_to_word.get(id).cloned();
            }
        }

        best_word
    }
}



/// Кодировщик: преобразует текст в сжатый тернарный вектор размерности VECTOR_DIM
pub struct Encoder<'a> {
    pub vocab: &'a VocabTable,
}

impl<'a> Encoder<'a> {
    pub fn new(vocab: &'a VocabTable) -> Self {
        Self { vocab }
    }

    pub fn encode(&self, text: &str) -> Vec<i8> {
        let mut aggregated = vec![0i32; VECTOR_DIM];
        let words: Vec<&str> = text
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty())
            .collect();

        if words.is_empty() {
            return vec![0; VECTOR_DIM];
        }

        for word in &words {
            let clean = word.to_lowercase();
            if let Some(&id) = self.vocab.word_to_id.get(&clean) {
                if let Some(v) = self.vocab.word_vectors.get(&id) {
                    for i in 0..VECTOR_DIM {
                        aggregated[i] += v[i] as i32;
                    }
                }
            } else {
                // Если слово новое/незнакомое, генерируем временный хэш-вектор
                let mut h: u64 = 0xcbf29ce484222325;
                for &b in clean.as_bytes() {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
                for i in 32..VECTOR_DIM {
                    h = h.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let val = ((h >> 33) % 3) as i32 - 1;
                    aggregated[i] += val;
                }
            }
        }

        // Пороговая активация в тернарный вектор
        aggregated
            .into_iter()
            .map(|sum| threshold_activation(sum, 0))
            .collect()
    }
}

/// Раскодировщик: преобразует кодовый вектор обратно в текст через Vocab Table
pub struct Decoder<'a> {
    pub vocab: &'a VocabTable,
}

impl<'a> Decoder<'a> {
    pub fn new(vocab: &'a VocabTable) -> Self {
        Self { vocab }
    }

    /// Раскодирование по кодовому вектору ответа и активированному эксперту
    pub fn decode_with_category(
        &self,
        code_vector: &[i8],
        query_vector: &[i8],
        primary_category: Option<usize>,
    ) -> String {
        if let Some(cat) = primary_category {
            // 1. Поиск ближайшей диалоговой реплики среди обученных пар этого эксперта
            if let Some(dialogs) = self.vocab.expert_dialog_responses.get(&cat) {
                let mut best_score = i32::MIN;
                let mut best_ans = None;

                for (q_vec, ans_text) in dialogs {
                    let mut score: i32 = 0;
                    for i in 0..VECTOR_DIM.min(q_vec.len()).min(query_vector.len()) {
                        if q_vec[i] == query_vector[i] && q_vec[i] != 0 {
                            score += 2;
                        } else if q_vec[i] == -query_vector[i] && q_vec[i] != 0 {
                            score -= 2;
                        }
                    }
                    if score > best_score {
                        best_score = score;
                        best_ans = Some(ans_text.clone());
                    }
                }

                // Если сходство с репликой высокое
                if let Some(ans) = best_ans {
                    if best_score > 4 {
                        return ans;
                    }
                }
            }

            // 2. Шаблон по умолчанию для категории
            if let Some(template) = self.vocab.category_templates.get(&cat) {
                return template.clone();
            }
        }
        self.decode(code_vector)
    }

    pub fn decode(&self, code_vector: &[i8]) -> String {
        if let Some(word) = self.vocab.find_closest_word(code_vector) {
            word
        } else {
            "неизвестно".to_string()
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Узлы конвейера: Node 2, Node 3, Node 1
// ---------------------------------------------------------------------------

/// Узел 2 (Node 2): выделение смысла и действия из входного вектора (семантический фильтр)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node2 {
    pub neurons: Vec<Neuron>,
}

impl Node2 {
    pub fn new(dim: usize, _seed: &mut u64) -> Self {
        // Узел 2: фильтрация и прямая передача семантики действия
        let mut neurons = Vec::with_capacity(dim);
        for i in 0..dim {
            let mut n = Neuron::new(dim, 0);
            n.weights[i] = 1;
            neurons.push(n);
        }
        Self { neurons }
    }

    pub fn forward(&self, input: &[i8]) -> Vec<i8> {
        self.neurons.iter().map(|n| n.forward(input)).collect()
    }
}

/// Узел 3 (Node 3): выделение объектов и контекста
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node3 {
    pub neurons: Vec<Neuron>,
}

impl Node3 {
    pub fn new(dim: usize, _seed: &mut u64) -> Self {
        // Узел 3: сохранение контекста объектов
        let mut neurons = Vec::with_capacity(dim);
        for i in 0..dim {
            let mut n = Neuron::new(dim, 0);
            n.weights[i] = 1;
            neurons.push(n);
        }
        Self { neurons }
    }

    pub fn forward(&self, input: &[i8]) -> Vec<i8> {
        self.neurons.iter().map(|n| n.forward(input)).collect()
    }
}

/// Узел 1 (Node 1): агрегация и выравнивание итогового кодового вектора ответа
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node1 {
    pub neurons: Vec<Neuron>,
}

impl Node1 {
    pub fn new(dim: usize, _seed: &mut u64) -> Self {
        // Узел 1 нормализует и очищает тернарный код отклика экспертов
        let mut neurons = Vec::with_capacity(dim);
        for i in 0..dim {
            let mut n = Neuron::new(dim, 0);
            n.weights[i] = 1;
            neurons.push(n);
        }
        Self { neurons }
    }

    pub fn forward(&self, input: &[i8]) -> Vec<i8> {
        self.neurons.iter().map(|n| n.forward(input)).collect()
    }
}

// ---------------------------------------------------------------------------
// 5. Иерархическое дерево маршрутизации (Main Router & Sub-Routers)
// ---------------------------------------------------------------------------

/// Подмаршрутизатор второго уровня (Sub-Router).
/// Управляет двумя группами по 5 мини-ИИ (10 экспертов).
/// Выбирает активных исполнителей внутри своей подгруппы.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubRouter {
    pub id: usize,
    /// Смещение ID экспертов (например, 0 для 0..9, 10 для 10..19 и т.д.)
    pub base_expert_id: usize,
    /// Нейроны селекции эксперта (по 1 нейрону на каждого из 10 экспертов)
    pub selector_neurons: Vec<Neuron>,
}

impl SubRouter {
    pub fn new(id: usize, base_expert_id: usize, input_dim: usize, seed: &mut u64) -> Self {
        let mut selector = Vec::with_capacity(EXPERTS_PER_SUB_ROUTER);
        for _ in 0..EXPERTS_PER_SUB_ROUTER {
            selector.push(Neuron::random(input_dim, 0, seed));
        }
        Self {
            id,
            base_expert_id,
            selector_neurons: selector,
        }
    }

    /// Выбор наиболее подходящего мини-ИИ в рамках этого подмаршрутизатора
    pub fn select_expert(&self, input: &[i8]) -> usize {
        let mut best_score = i32::MIN;
        let mut best_idx = 0;

        for (idx, neuron) in self.selector_neurons.iter().enumerate() {
            // Вычисляем скор как сумму совпадений связей без умножения
            let mut sum: i32 = 0;
            for i in 0..neuron.weights.len().min(input.len()) {
                match neuron.weights[i] {
                    1 => sum += input[i] as i32,
                    -1 => sum -= input[i] as i32,
                    _ => {}
                }
            }
            if sum > best_score {
                best_score = sum;
                best_idx = idx;
            }
        }
        self.base_expert_id + best_idx
    }
}

/// Главный распределитель (Main Router).
/// Получает объединенные сигналы от Node 2 и Node 3,
/// активирует 1 или 2 ветки из 4 Sub-Router'ов (остальные не вычисляются).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MainRouter {
    pub sub_router_selectors: Vec<Neuron>,
    pub sub_routers: Vec<SubRouter>,
}

impl MainRouter {
    pub fn new(seed: &mut u64) -> Self {
        let mut selectors = Vec::with_capacity(NUM_SUB_ROUTERS);
        for _ in 0..NUM_SUB_ROUTERS {
            selectors.push(Neuron::random(VECTOR_DIM, 0, seed));
        }

        let mut sub_routers = Vec::with_capacity(NUM_SUB_ROUTERS);
        for i in 0..NUM_SUB_ROUTERS {
            sub_routers.push(SubRouter::new(i, i * EXPERTS_PER_SUB_ROUTER, VECTOR_DIM, seed));
        }

        Self {
            sub_router_selectors: selectors,
            sub_routers,
        }
    }

    /// Выбор 1 или 2 активных веток подмаршрутизаторов.
    /// Главный роутер опрашивает свои 4 ветки (Sub-Routers) по максимальному совпадению с их группами экспертов
    pub fn route(&self, input: &[i8]) -> Vec<usize> {
        let mut scores: Vec<(usize, i32)> = Vec::new();

        for (idx, sub_router) in self.sub_routers.iter().enumerate() {
            // Максимальный скор среди экспертов данного подмаршрутизатора
            let mut max_expert_score = i32::MIN;
            for neuron in &sub_router.selector_neurons {
                let mut sum: i32 = 0;
                for i in 0..neuron.weights.len().min(input.len()) {
                    match neuron.weights[i] {
                        1 => sum += input[i] as i32,
                        -1 => sum -= input[i] as i32,
                        _ => {}
                    }
                }
                if sum > max_expert_score {
                    max_expert_score = sum;
                }
            }
            scores.push((idx, max_expert_score));
        }

        // Сортируем по убыванию соответствия
        scores.sort_by(|a, b| b.1.cmp(&a.1));

        let mut active = vec![scores[0].0];
        if scores.len() > 1 && (scores[0].1 - scores[1].1) <= 2 {
            active.push(scores[1].0);
        }
        active
    }
}

// ---------------------------------------------------------------------------
// 6. Память / Буфер кэша и Диспетчер (Cache Buffer & Dispatcher)
// ---------------------------------------------------------------------------

/// Запись в буфере кэша
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub query: String,
    pub query_code: Vec<i8>,
    pub response: String,
    pub response_code: Vec<i8>,
}

/// Память / Буфер кэша (двусторонняя связь)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheBuffer {
    pub entries: Vec<CacheEntry>,
}

impl CacheBuffer {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Поиск точного или близкого ответа по Хэммингу/тернарному совпадению
    pub fn lookup(&self, query: &str, query_code: Option<&[i8]>) -> Option<&CacheEntry> {
        let clean = query.trim().to_lowercase();
        // 1. Точный поиск по строке
        for entry in self.entries.iter().rev() {
            if entry.query.to_lowercase() == clean {
                return Some(entry);
            }
        }

        // 2. Близкий поиск по тернарному коду (только при полном совпадении вектора запроса)
        if let Some(code) = query_code {
            for entry in self.entries.iter().rev() {
                let mut diff = 0;
                for i in 0..VECTOR_DIM.min(code.len()).min(entry.query_code.len()) {
                    if code[i] != entry.query_code[i] {
                        diff += 1;
                    }
                }
                if diff == 0 {
                    return Some(entry);
                }
            }
        }

        None
    }

    pub fn insert(&mut self, query: String, query_code: Vec<i8>, response: String, response_code: Vec<i8>) {
        // Ограничиваем размер кэша
        if self.entries.len() >= 1000 {
            self.entries.remove(0);
        }
        self.entries.push(CacheEntry {
            query,
            query_code,
            response,
            response_code,
        });
    }

    /// Очистка кэш-буфера
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Диспетчер (Dispatcher)
pub struct Dispatcher {
    pub cache: CacheBuffer,
}

impl Dispatcher {
    pub fn new() -> Self {
        Self {
            cache: CacheBuffer::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// 7. Спецификация бинарного формата .tyai
// ---------------------------------------------------------------------------

/// Заголовок бинарного файла .tyai
#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TyaiHeader {
    /// Магические байты b"TYAI"
    pub magic: [u8; 4],
    /// Версия формата
    pub version: u32,
    /// Общее количество модулей (40 экспертов + узлы + роутеры + словарь)
    pub total_modules: u32,
    /// Количество записей в Index Table
    pub index_count: u32,
    /// Смещение до начала Index Table в файле
    pub index_offset: u64,
}

/// Запись в таблице индексов (Index Table)
#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TyaiIndexEntry {
    /// Идентификатор модуля (0..39: эксперты, 100: Vocab, 101: Node1, 102: Node2, 103: Node3, 104: MainRouter)
    pub module_id: u32,
    /// Смещение полезной нагрузки в байтах
    pub offset: u64,
    /// Размер полезной нагрузки в байтах
    pub size: u64,
}

pub const MODULE_ID_VOCAB: u32 = 100;
pub const MODULE_ID_NODE1: u32 = 101;
pub const MODULE_ID_NODE2: u32 = 102;
pub const MODULE_ID_NODE3: u32 = 103;
pub const MODULE_ID_ROUTER: u32 = 104;

// ---------------------------------------------------------------------------
// 8. Безградиентное эволюционное обучение (Эволюция / Мутации)
// ---------------------------------------------------------------------------

/// Функция вычисления ошибки (дистанция Хэмминга / несоответствие кодов)
pub fn compute_error(predicted: &[i8], target: &[i8]) -> i32 {
    let mut err = 0;
    for i in 0..predicted.len().min(target.len()) {
        if predicted[i] != target[i] {
            err += 1;
        }
    }
    err
}

/// Мутация тернарных весов нейрона с заданной вероятностью (1 из 10)
pub fn mutate_neuron(neuron: &mut Neuron, seed: &mut u64) {
    for w in &mut neuron.weights {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        // ~10% шанс мутации
        if ((*seed >> 32) % 10) == 0 {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let delta = ((*seed >> 33) % 3) as i8 - 1; // -1, 0, 1
            *w = clamp_ternary(*w + delta);
        }
    }
    // Редкая мутация порога
    if ((*seed >> 28) % 20) == 0 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let delta = ((*seed >> 32) % 3) as i32 - 1;
        neuron.threshold += delta;
    }
}

/// Мутация всей структуры Мини-ИИ
pub fn mutate_mini_ai(ai: &MiniAI, seed: &mut u64) -> MiniAI {
    let mut candidate = ai.clone();
    for neuron in &mut candidate.hidden_neurons {
        mutate_neuron(neuron, seed);
    }
    for neuron in &mut candidate.output_neurons {
        mutate_neuron(neuron, seed);
    }
    candidate
}

/// Обучение Мини-ИИ на датасете через эволюционный авторост (до 500-1000 нейронов) и быструю локальную подстройку
pub fn train_mini_ai(
    ai: &mut MiniAI,
    dataset: &[(&[i8], &[i8])],
    target_neurons: usize,
    seed: &mut u64,
) {
    if dataset.is_empty() {
        return;
    }

    // 1. Конструктивный нейрогенез: автовыращиваем нейроны от 10 до целевого количества (500-1000)
    while ai.hidden_neurons.len() < target_neurons && ai.hidden_neurons.len() < ai.max_neurons {
        let chunk = 50.min(target_neurons - ai.hidden_neurons.len());
        ai.spawn_neurons(chunk, seed);
    }

    // 2. Локальная безградиентная подстройка тернарных связей (Ассоциативное тернарное обучение Hebb/Delta)
    // Для каждого сэмпла из сотен примеров активируем скрытые нейроны и настраиваем выходные веса:
    // w_ij <- clamp(w_ij + h_j * target_i)
    for &(inp, target) in dataset {
        let mut hidden_acts = Vec::with_capacity(ai.hidden_neurons.len());
        for h in &ai.hidden_neurons {
            hidden_acts.push(h.forward(inp));
        }

        // Обновляем веса выходных нейронов
        for (i, out_neuron) in ai.output_neurons.iter_mut().enumerate() {
            let t = if i < target.len() { target[i] } else { 0 };
            if t == 0 {
                continue;
            }
            let len = out_neuron.weights.len().min(hidden_acts.len());
            for j in 0..len {
                let h_act = hidden_acts[j];
                if h_act != 0 {
                    // Синаптическая пластичность без умножений
                    let delta = if h_act == t { 1 } else { -1 };
                    out_neuron.weights[j] = clamp_ternary(out_neuron.weights[j] + delta);
                }
            }
        }
    }

    // 3. Быстрая эволюционная полировка лучших тернарных весов
    let evaluate = |model: &MiniAI| -> i32 {
        let mut total_loss = 0;
        for &(inp, target) in dataset {
            let out = model.forward(inp);
            total_loss += compute_error(&out, target);
        }
        total_loss
    };

    let mut best_loss = evaluate(ai);
    for _ in 0..10 {
        if best_loss == 0 {
            break;
        }
        let candidate = mutate_mini_ai(ai, seed);
        let loss = evaluate(&candidate);
        if loss < best_loss {
            best_loss = loss;
            *ai = candidate;
        }
    }

    // 4. Обрезка связей (Pruning), сохраняя не менее 500 нейронов
    ai.prune();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_and_index_sizes() {
        let header = TyaiHeader {
            magic: *TYAI_MAGIC,
            version: TYAI_VERSION,
            total_modules: 45,
            index_count: 45,
            index_offset: 24,
        };
        let h_bytes = bincode::serialize(&header).unwrap();
        println!("Header size_of: {}, serialized: {}", std::mem::size_of::<TyaiHeader>(), h_bytes.len());

        let entry = TyaiIndexEntry { module_id: 1, offset: 100, size: 200 };
        let e_bytes = bincode::serialize(&entry).unwrap();
        println!("Index entry size_of: {}, serialized: {}", std::mem::size_of::<TyaiIndexEntry>(), e_bytes.len());
    }
}


