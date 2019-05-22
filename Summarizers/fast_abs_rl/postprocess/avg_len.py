import os
OUTPUT_DIR = "../preprocess/finished_files/output/"

avg_len = 0.0

for fname in os.listdir(OUTPUT_DIR):
    avg_len += len(open(OUTPUT_DIR + fname).read().split())

print(avg_len / len(os.listdir(OUTPUT_DIR)))
