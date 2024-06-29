from langchain_core.prompts import PromptTemplate


def prompt1(sent, query):
    """
    You are a helpful assistant. 
    Given the following sentence: التقيت مترجم الرئيس المشهور الذي يشارك في فعاليات المؤتمر اليوم. 
    Answer the following question with only one word: من يشارك في المنتدى؟
    """
    prompt_template_one = PromptTemplate.from_template(
        "You are a helpful assistant. Given the following sentence: {sentence} Answer the following question with only one word: {question}"
    )
    template1 = prompt_template_one.format(sentence=sent, question=query)
    return template1

def prompt2(sent, query, choices):
    """
    You are a helpful assistant. Given the following sentence: التقيت مترجم الرئيس المشهور الذي يشارك في فعاليات المؤتمر اليوم. 
    Answer the following question: من يشارك في المنتدى؟ 
    Choose one choice without justification: المترجم أم الرئيس
    """
    prompt_template_two = PromptTemplate.from_template(
        "You are a helpful assistant. Given the following sentence: {sentence} Answer the following question: {question} Choose one choice without justification: {choices}"
    )
    template2 = prompt_template_two.format(sentence=sent, question=query, choices=choices)
    return template2

