GEN_SYS_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request."
)

PRE_SYS_PROMPT = (
    "You are a helpful assistant that performs time series prediction. "
    "The user will provide a sequence and you will predict the sequence."
)

ANA_SYS_PROMPT = (
    "You are a helpful assistant that performs time series analysis. "
    "The user will provide a sequence and you will respond to the questions "
    "based on this sequence."
)

PRE_INST_PROMPT = "Please predict the following sequence carefully."

PRE_INST_PROMPT_TEXT = (
    "Please predict the following sequence carefully. "
    "Context knowledge you may consider: {}"
)

ANA_INST_PROMPT_TEXT = (
    "Please answer the following question carefully after analyzing the sequence: {}"
)

TEMPLATE = """{}

#### Instruction:

{}

#### Input:

{}

#### Response:

{}"""


def getPrompt(
    flag,
    instruction=None,
    input=None,
    response=None,
    context=None,
):
    if flag == "general":
        system = GEN_SYS_PROMPT

        if instruction is None:
            raise ValueError(
                "Instruction must be provided for general tasks."
            )

        instruction_text = str(instruction).strip()
        input_text = "" if input is None else str(input).strip()

    elif flag == "prediction":
        system = PRE_SYS_PROMPT

        if input is None:
            raise ValueError(
                "Input must be provided for prediction tasks."
            )

        input_text = str(input).strip()

        context_text = (
            ""
            if context is None
            else str(context).strip()
        )

        if context_text:
            instruction_text = PRE_INST_PROMPT_TEXT.format(
                context_text
            )
        else:
            instruction_text = PRE_INST_PROMPT

    elif flag == "analysis":
        system = ANA_SYS_PROMPT

        if instruction is None:
            raise ValueError(
                "Instruction must be provided for analysis tasks."
            )

        if input is None:
            raise ValueError(
                "Input must be provided for analysis tasks."
            )

        instruction_text = ANA_INST_PROMPT_TEXT.format(
            str(instruction).strip()
        )
        input_text = str(input).strip()

    else:
        raise ValueError(
            "Flag must be one of "
            "'general', 'prediction', or 'analysis'."
        )

    response_text = (
        ""
        if response is None
        else str(response).strip()
    )

    return TEMPLATE.format(
        system,
        instruction_text,
        input_text,
        response_text,
    )