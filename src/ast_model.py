from transformers import ASTConfig, ASTForAudioClassification


def ASTModel(num_labels):
    config = ASTConfig.from_pretrained(
        "MIT/ast-finetuned-speech-commands-v2",
        num_labels=num_labels
    )
    return ASTForAudioClassification(config) 