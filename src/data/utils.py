import os
import io
import zstandard as zstd
import pandas as pd


# loading the data
def create_dataframe_from_csv_zst(filepath: str) -> pd.DataFrame:
    dctx = zstd.ZstdDecompressor()
    with open(filepath, 'rb') as compressed:
        with dctx.stream_reader(compressed) as reader:
            decompressed = io.TextIOWrapper(reader, encoding='utf-8')
            df = pd.read_csv(
                decompressed,
                low_memory=True
            )
    return df


# reading a large csv file in chunks
def read_large_csv_zst(file_path, chunksize=100000, **kwargs):
    dctx = zstd.ZstdDecompressor()
    with open(file_path, 'rb') as fh:
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            chunk_iter = pd.read_csv(text_stream, chunksize=chunksize, **kwargs)
            for chunk in chunk_iter:
                yield chunk



# compressing the dataset
def compress_dataframe_to_zst(df: pd.DataFrame, output_file: str, compression_level: int = 3) -> None:
    cctx = zstd.ZstdCompressor(level=compression_level)
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    compressed_data = cctx.compress(csv_bytes)
    with open(output_file, 'wb') as f:
        f.write(compressed_data)
    print(f"Compressed to file {output_file} with size {len(compressed_data)} bytes")

