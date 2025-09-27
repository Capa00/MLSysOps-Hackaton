import pandas as pd

def main():
    complete_dataset = pd.read_csv("./data/Export Data-data-as-joinbyfield-2025-09-26 21_04_20.csv")
    cols = [c for c in complete_dataset.columns if "frame_classify_latency" in c]
    complete_dataset["frame_classify_latency"] = complete_dataset[cols].bfill(axis=1).iloc[:, 0]

    df = complete_dataset[["frame_classify_latency"]].copy()


    df["fps"] = 1000 / df["frame_classify_latency"]

if __name__ == '__main__':
    main()

