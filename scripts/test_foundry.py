from foundry_local_sdk import Configuration, FoundryLocalManager


def main():
    print("Initializing Foundry Local...")

    FoundryLocalManager.initialize(
        Configuration(app_name="Recall")
    )
    manager = FoundryLocalManager.instance

    print("Selecting model...")
    model = manager.catalog.get_model("qwen2.5-0.5b")

    print("Downloading model...")
    model.download(
        lambda p: print(
            f"\rDownloading: {p:.1f}%",
            end="",
            flush=True
        )
    )
    print()

    print("Loading model...")
    model.load()

    print("Model loaded.")

    client = model.get_chat_client()

    messages = [
        {
            "role": "user",
            "content": (
                "Explain semantic search in three short sentences."
            )
        }
    ]

    print("\nMODEL RESPONSE:\n")

    response = client.complete_chat(messages)
    print(response.choices[0].message.content)

    model.unload()

    print("\nModel unloaded successfully.")


if __name__ == "__main__":
    main()