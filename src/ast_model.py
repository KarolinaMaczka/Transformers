from transformers import ASTConfig, ASTForAudioClassification

def ASTModel(
    num_labels: int,
    **config_kwargs
) -> ASTForAudioClassification:
    config = ASTConfig.from_pretrained(
        "MIT/ast-finetuned-speech-commands-v2",
        num_labels=num_labels
    )
    for key, value in config_kwargs.items():
        if value is not None and hasattr(config, key):
            print(f'Setting attribute {key} to {value}')
            setattr(config, key, value)
            
    return ASTForAudioClassification(config)
