def has_tied_weights(model_type: str) -> bool:
    """
    Checks if a given Hugging Face transformers model_type string employs
    tied weights. The Gemma family of models is currently detected.
    GLM models typically do not use tied weights.

    The model type is typically obtained from the `config.model_type`
    attribute after loading a model configuration (e.g., using AutoConfig).

    Args:
        model_type (str): The model type string (e.g., 'gemma', 'gemma2', 'llama', 'glm').

    Returns:
        bool: True if the model is of a lineage which employs tied weights,
        False otherwise.
    """
    if not isinstance(model_type, str):
        return False

    # Standard model types used for Gemma and related architectures in Hugging Face
    # 'gemma': Original Gemma 1
    # 'gemma2': Gemma 2 models
    # 'gemma3': Gemma 3 models
    # 'paligemma': PaliGemma Vision-Language Models (VLM)
    gemma_family_types = {"gemma", "gemma2", "gemma3", "paligemma"}
    
    # GLM models (ChatGLM, GLM-4, etc.) typically do not use tied weights
    # but we include them here for completeness and future compatibility
    glm_family_types = {"chatglm", "glm"}
    
    # GPT-OSS models (GPT-OSS-20B, etc.) typically do not use tied weights
    gpt_oss_family_types = {"gptoss"}

    # The check is case-insensitive for robustness
    model_type_lower = model_type.lower()
    
    # Check if it's a Gemma family model (uses tied weights)
    if model_type_lower in gemma_family_types:
        return True
    
    # GLM models typically don't use tied weights, but check for future variants
    # For now, return False for GLM models
    if model_type_lower in glm_family_types or "glm" in model_type_lower:
        return False
    
    # GPT-OSS models typically don't use tied weights
    if model_type_lower in gpt_oss_family_types or "gptoss" in model_type_lower:
        return False
    
    return False