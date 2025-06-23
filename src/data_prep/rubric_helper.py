from llm import get_completion, Models
import logging
import jinja2 as j2

REQUEST_TO_CONTINUE_QUESTIONS = "Thanks for your patience so far. I would like to ask you a few questions to better understand your situation. Is that OK?"
def current_diagnosis(conversation, prompts_loader, template_file='diagnosis_prompt.j2', model=Models.GEMINI_PRO_MODEL.value):
    # Get the current diagnosis from the conversation
    diagnosis_prompt = j2.Environment(loader=prompts_loader).get_template(template_file).render(conversation=conversation[1:])
    return get_completion(messages=[{'role':'system', 'content': diagnosis_prompt}, {'role': 'user', 'content':'Diagnosis please.'}], 
                          model=model, temperature=0)

def get_rubric_label_prompt(conversation, prompts_loader, template_file='rubric_prompt.j2'):
    # Load the rubric label prompt from the file
    rubric_label_prompt = j2.Environment(loader=prompts_loader).get_template(template_file).render(conversation=conversation[1:])
    return rubric_label_prompt

def create_rubrics(training_conversations, metadata, prompts_loader):
    rubric = {}
    for cid, training_conversation in training_conversations.items():
        type_ = metadata[cid]["type"]
        if type_ == "early_exit":
            diagnosis = current_diagnosis(training_conversation, prompts_loader, template_file='diagnosis_prompt.j2')
            rubric[cid] = diagnosis
        elif type_ == "check_ins":
            rubric[cid] = REQUEST_TO_CONTINUE_QUESTIONS
        elif type_ == "standard":
            logging.info("Generating rubric label for conversation ID: %s", cid)
            prompt = get_rubric_label_prompt(conversation=training_conversation[1:], prompts_loader=prompts_loader)
            response = get_completion(messages=[{"role": "user", "content": prompt}], 
                                      model=Models.GEMINI_PRO_MODEL.value, temperature=0, is_chat=True)
            rubric[cid] = response

    return rubric
