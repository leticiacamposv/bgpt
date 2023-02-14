import streamlit as st
import torch
from transformers import pipeline, GPTJForCausalLM, AutoTokenizer

# Page config
st.set_page_config(page_title='BGPT-J', page_icon=':eyeglasses:', layout='wide')

# Primary title on page
st.title('BGPT-J')

# Sidebar
st.sidebar.markdown("Selecione os parâmetros")
temperature = st.sidebar.slider('Temperature', min_value=0.00, max_value=1.00, step=0.01, help='Controla a aleatoriedade do texto gerado. Quanto maior a temperatura, mais "criativo" e arriscad será o modelo. ')
max_length = st.sidebar.slider('Max length', min_value=1, max_value=10000, step=1, help='Quantidade máxima de tokens ou "limite de palavras" da resposta. Quanto maior o número de tokens, mais demorada a resposta')
top_p = st.sidebar.slider('Top P', min_value=0.00, max_value=1.00, step=0.01, help='Grau de consideração de inclusão de palavras com probabilidades menores aparecerem no texto. Também controla a criatividade')
freq_penalty = st.sidebar.slider('Frequency penalty', min_value=0.00, max_value=2.00, step=0.01, help='Grau de penalidade para repetição da mesma palavra em texto')
epsilon_cutoff =st.sidebar.slider('Epsilon cutoff', min_value=0.000, max_value=1.000, step=0.001, help='Determina um limite mínimo de probabilidade para os tokens a serem usados. Ex: epsilon > 0.7 -> Apenas tokens com mais de 70 porcento de probabilidade serão impressos')
best_of = st.sidebar.slider('Best of', min_value=0.00, max_value=20.00, step=1.0, help='Quantidade de respostas diferentes geradas. Use para efeito de variedade na resposta')
#Inject start text
prob = st.sidebar.checkbox('Mostrar probabilidade dos tokens')

#Playground
prompt = st.text_area('Digite aqui a sua pergunta/pedido', height=200)
btn_submit = st.button('Enviar')

#Modelo
model = torch.load("gptj8bit.pt")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

#Predict
if btn_submit:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        max_length=max_length,
        top_p=top_p,
        epsilon_cutoff=epsilon_cutoff, 
        repetition_penalty=freq_penalty,
        num_return_sequences=best_of,
        output_scores=prob,
        temperature=temperature)

    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    st.write(gen_text)