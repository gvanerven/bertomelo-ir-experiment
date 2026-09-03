use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use serde::Serialize;
use std::fs::File;
use std::io::{BufWriter, Write};

#[derive(Serialize)]
struct WordRecord {
    count_words: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Abre o arquivo Parquet
    let parquet_file = File::open("/home/gvanerven/code/doutorado/bertomelo-ir-experiment/data/ds_select_size.parquet")?;
    let reader = SerializedFileReader::new(parquet_file)?;
    
    // Obtém o número total de linhas a partir dos metadados
    let metadata = reader.metadata();
    let num_rows = metadata.file_metadata().num_rows();

    // 2. Prepara o arquivo NDJSON para escrita com buffer (muito mais rápido que o f.flush() por linha)
    let output_file = File::create("/home/gvanerven/code/doutorado/bertomelo-ir-experiment/data/count_words.ndjson")?;
    let mut writer = BufWriter::new(output_file);

    let mut count_words: usize = 0;

    // 3. Itera sobre os registros (rows) do arquivo Parquet
    let row_iter = reader.get_row_iter(None)?;

    for row in row_iter {
        let row = row?;
        
        // Obtém o campo "text" da linha atual
        if let Ok(text) = row.get_string(0) {
            let l = text.split_whitespace().count();
            count_words += l;

            // Escreve a contagem no formato NDJSON
            let record = WordRecord { count_words: l };
            serde_json::to_writer(&mut writer, &record)?;
            writer.write_all(b"\n")?;
        }
    }

    // Garante que todo o buffer restante seja gravado no disco ao finalizar
    writer.flush()?;

    // 4. Imprime os resultados
    println!("Total: {}", count_words);
    if num_rows > 0 {
        println!("Average: {}", count_words as f64 / num_rows as f64);
    }

    Ok(())
}