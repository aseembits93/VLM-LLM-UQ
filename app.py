import gradio as gr
import spaces
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from PIL import Image
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images
import torch
import pickle
import numpy as np
'''
What is this?
A. Face
B. Cat
C. Dog
D. Box
E. I don’t know
F. None of the above
'''
ALL_OPTIONS = ['A', 'B', 'C', 'D', 'E', 'F']
OPTIONS = ['A', 'B', 'C', 'D']#, 'E', 'F']

TITLE = "# [VLM Uncertainty Demo](https://github.com/aseembits93/VLM-LLM-UQ)"
DESCRIPTION = "Quantify the uncertainty of VQA models via Conformal Prediction"
alpha=0.1
css = """
  #output {
    height: 500px; 
    overflow: auto; 
    border: 1px solid #ccc; 
  }
"""
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_qhat(result_data, alpha):
    """
    Apply conformal prediction to obtain sets of predicted answers on each instance based on its softmax scores.
    Here the APS score function is utilized.
    """
    cal_scores = []
    for row in result_data:
        probs = softmax(row["logits"][:-1])
        truth_answer = row["answer"]
        cal_pi = np.argsort(probs)[::-1] # descending order
        cal_sum = np.take_along_axis(probs, cal_pi, axis=0).cumsum()
        cal_sum_r = np.take_along_axis(cal_sum, cal_pi.argsort(), axis=0)
        cal_score = cal_sum_r[ALL_OPTIONS.index(truth_answer)]
        cal_scores.append(cal_score)
    n = len(result_data)
    q_level = np.ceil((n+1) * (1-alpha)) / n
    qhat = np.quantile(cal_scores, q_level, method='higher')
    return qhat

#import subprocess
#subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)
disable_torch_init()
model_name = get_model_name_from_path('liuhaotian/llava-v1.5-7b')
tokenizer, model, image_processor, context_len = load_pretrained_model('liuhaotian/llava-v1.5-7b', None, model_name)
model.cuda()
file_name = f'mmbench.pkl'
with open(file_name, 'rb') as f:
    result_data = pickle.load(f)
qhat = get_qhat(result_data, alpha)

def get_inputs_custom(row, tokenizer, image_processor, model, image):
    question_text = row['question']
    qs = question_text
    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
    image_size = image.size
    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv = conv_templates['vicuna_v1'].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    image_tensor = process_images([image], image_processor, model.config)[0]
    return {
        'input_ids': input_ids.unsqueeze(0).cuda(),
        'images': image_tensor.unsqueeze(0).half().cuda(),
        'image_sizes': [image_size]
    }

def get_prediction_set(logits, qhat, alpha):
    pred_sets = {}
    probs = softmax(logits)
    cal_pi = np.argsort(probs)[::-1] # descending order
    cal_sum = np.take_along_axis(probs, cal_pi, axis=0).cumsum()
    ps = []
    ii = 0
    while ii < len(cal_sum) and cal_sum[ii] <= qhat:
        op_id = cal_pi[ii]
        ps.append(ALL_OPTIONS[op_id])
        ii += 1
    if len(ps) == 0:
        op_id = cal_pi[ii]
        ps.append(ALL_OPTIONS[op_id])
    return ps
    
@spaces.GPU
def run_example(prompt, image, qhat, alpha):
    row = {'index': 0, 'question': '', 'hint': '', 'A': 'Dog', 'B': 'Human', 'C': 'Cat', 'D': 'Cow', 'E':'','F':'','answer': ''}
    '''
    'index': 487, 'question': 'Which term matches the picture?', 'hint': 'Read the text.\nThe sea is home to many different groups, or phyla, of animals. Two of these are cnidarians and echinoderms.\nCnidarian comes from a Greek word that means "nettle," a stinging type of plant. Cnidarians have tentacles all around their mouths, which they use to sting prey and pull the prey toward their mouths.\nEchinoderm comes from Greek words meaning "spiny" and "skin." Echinoderms have stiff bodies, and their spines may stick out of their skins. Adult echinoderms\' bodies are often arranged in five balanced parts, like a star.', 'A': 'echinoderm', 'B': 'cnidarian', 'C': 'Solution B', 'D': 'The keyboard is touching the cat.', 'answer': 'B', 'category': 'attribute_recognition', 'source': 'scienceqa', 'l2-category': 'finegrained_perception (instance-level)', 'split': 'dev', 'image_path': 'datasets/mmbench/images/487.png', 'id': 487
    '''
    row['question']=prompt
    inputs = get_inputs_custom(row, tokenizer, image_processor, model, image)
    option_ids = [tokenizer.encode(opt)[-1] for opt in ALL_OPTIONS]
    with torch.no_grad():
        output = model(
            **inputs,
            return_dict=True,
        )
    logits = output.logits.detach().cpu().numpy()[:, -1, :]
    logits = logits[0,option_ids]
    pred_set = get_prediction_set(logits, qhat, alpha)
    #return ALL_OPTIONS[logits[0,option_ids].argmax().item()]
    return pred_set

def process_image(image, text_input):
    image = Image.fromarray(image,mode='RGB') 
    results = run_example(text_input, image, qhat, alpha)
    return results


with gr.Blocks(css=css) as demo:
    gr.Markdown(TITLE)
    gr.Markdown(DESCRIPTION)
    with gr.Tab(label="VLM Uncertainty Demo"):
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(label="Input Picture")
                text_input = gr.Textbox(label="Text Input")
                submit_btn = gr.Button(value="Submit")
            with gr.Column():
                output_text = gr.Textbox(label="Output Text")
        gr.Examples(
            examples=[
                ["yogamat.webp", 'What is this?\nA. Face\nB. Cat\nC. Yoga Mat\nD. Box\nE. I don’t know\nF. None of the above'],
                ["dog.webp", 'What is this?\nA. Dog\nB. Cat\nC. Yoga Mat\nD. Box\nE. I don’t know\nF. None of the above'],
                ["cat.jpg", 'What is this?\nA. Face\nB. Cat\nC. Yoga Mat\nD. Box\nE. I don’t know\nF. None of the above'],
            ],
            inputs=[input_img, text_input],
            outputs=[output_text],
            fn=process_image,
            cache_examples=True,
            label='Try the examples below'
        )
        submit_btn.click(process_image, [input_img, text_input], [output_text])

demo.launch(debug=True,share=True)
