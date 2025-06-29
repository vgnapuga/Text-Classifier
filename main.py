from src.pipeline import TextClassificationPipeline


def main() -> None:
    pipeline = TextClassificationPipeline("data/data.csv")
    pipeline.run(True)


if (__name__ == "__main__"):
    main()