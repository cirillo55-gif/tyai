use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use tyai_system::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== [packer] Упаковка модулей в единый бинарный контейнер model.tyai ===");

    // Список модулей для упаковки:
    // ID 0..39: 40 мини-ИИ экспертов
    // ID 100: Vocab Table
    // ID 101: Node 1
    // ID 102: Node 2
    // ID 103: Node 3
    // ID 104: Main & Sub-Routers
    let mut files_to_pack: Vec<(u32, String)> = Vec::new();

    for id in 0..TOTAL_EXPERTS {
        files_to_pack.push((id as u32, format!("brain_raw/expert_{:02}.bin", id)));
    }
    files_to_pack.push((MODULE_ID_VOCAB, "brain_raw/vocab.bin".to_string()));
    files_to_pack.push((MODULE_ID_NODE1, "brain_raw/node1.bin".to_string()));
    files_to_pack.push((MODULE_ID_NODE2, "brain_raw/node2.bin".to_string()));
    files_to_pack.push((MODULE_ID_NODE3, "brain_raw/node3.bin".to_string()));
    files_to_pack.push((MODULE_ID_ROUTER, "brain_raw/router.bin".to_string()));

    let total_modules = files_to_pack.len() as u32;

    // Спецификация формата .tyai:
    // [TyaiHeader] -> [Index Table (entries)] -> [Payload blocks]

    let header_size = bincode::serialized_size(&TyaiHeader {
        magic: *TYAI_MAGIC,
        version: TYAI_VERSION,
        total_modules: 0,
        index_count: 0,
        index_offset: 0,
    })?;

    let index_entry_size = bincode::serialized_size(&TyaiIndexEntry {
        module_id: 0,
        offset: 0,
        size: 0,
    })?;

    let index_table_size = index_entry_size * total_modules as u64;
    let payload_start_offset = header_size + index_table_size;

    let header = TyaiHeader {
        magic: *TYAI_MAGIC,
        version: TYAI_VERSION,
        total_modules,
        index_count: total_modules,
        index_offset: header_size,
    };

    let mut out_file = File::create("model.tyai")?;

    // Резервируем место под Header и Index Table
    let dummy_header = vec![0u8; header_size as usize];
    let dummy_index = vec![0u8; index_table_size as usize];
    out_file.write_all(&dummy_header)?;
    out_file.write_all(&dummy_index)?;

    let mut current_offset = payload_start_offset;
    let mut index_entries: Vec<TyaiIndexEntry> = Vec::new();

    println!("-> Запись блоков полезной нагрузки (Payload)...");
    for (module_id, file_path) in files_to_pack {
        let mut in_file = File::open(&file_path)
            .map_err(|e| format!("Не удалось открыть {}: {}", file_path, e))?;
        let mut data = Vec::new();
        in_file.read_to_end(&mut data)?;

        let size = data.len() as u64;
        out_file.write_all(&data)?;

        index_entries.push(TyaiIndexEntry {
            module_id,
            offset: current_offset,
            size,
        });

        current_offset += size;
    }

    // Возвращаемся в начало и записываем актуальный Header
    out_file.seek(SeekFrom::Start(0))?;
    let header_bytes = bincode::serialize(&header)?;
    // Проверим выравнивание / точный размер
    assert!(header_bytes.len() <= header_size as usize);
    out_file.write_all(&header_bytes)?;

    // Записываем Index Table
    out_file.seek(SeekFrom::Start(header.index_offset))?;
    for entry in &index_entries {
        let entry_bytes = bincode::serialize(entry)?;
        out_file.write_all(&entry_bytes)?;
    }

    println!("-> Сформирован файл: model.tyai");
    println!("   Магические байты: {:?}", std::str::from_utf8(&header.magic).unwrap());
    println!("   Версия формата:   v{}", header.version);
    println!("   Всего модулей:    {}", header.total_modules);
    println!("   Итоговый размер:  {} байт (~{:.2} КБ)", current_offset, current_offset as f64 / 1024.0);

    println!("\nУпаковка завершена успешно!\n");
    Ok(())
}
