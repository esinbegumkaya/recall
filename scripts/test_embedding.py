from foundry_local_sdk import Configuration, FoundryLocalManager


def main():
    print("Initializing Foundry Local...")

    FoundryLocalManager.initialize(
        Configuration(app_name="Recall")
    )

    manager = FoundryLocalManager.instance

    print("Selecting embedding model...")

    model = manager.catalog.get_model(
        "qwen3-embedding-0.6b"
    )

    print("Downloading embedding model...")

    model.download(
        lambda p: print(
            f"\rDownloading: {p:.1f}%",
            end="",
            flush=True,
        )
    )

    print("\nLoading model...")

    model.load()

    print("Model loaded.")

    embedding_client = model.get_embedding_client()

    text = "Recall is a private semantic search engine."

    response = embedding_client.generate_embedding(text)

    embedding = response.data[0].embedding

    print("\nEmbedding created.")
    print("Vector length:", len(embedding))
    print("First 10 values:")
    print(embedding[:10])

    model.unload()

    print("\nEmbedding model unloaded.")


if __name__ == "__main__":
    main()