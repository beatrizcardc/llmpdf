import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline
import os
import tempfile

# Configuração inicial
st.set_page_config(page_title="PDF Conversational Assistant", page_icon="📄", layout="wide")

# Barra lateral
with st.sidebar:
    st.header("Configurações")
    action = st.selectbox(
        "Escolha a ação:",
        ["Resumo", "Extração de trecho", "Perguntas e respostas"]
    )
    st.markdown("### Faça upload do arquivo")
    uploaded_file = st.file_uploader("Envie seu PDF ou TXT", type=["pdf", "txt"])
    save_download = st.checkbox("Salvar resultado para download", value=True)

@st.cache_resource
def load_llm():
    return pipeline("text2text-generation", model="t5-small", framework="pt")

# Função para ler arquivos PDF
def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Função para ler arquivos TXT
def read_txt(file):
    return file.read().decode("utf-8")

# Carregar modelo LLM (usando Ollama 3 ou similar)
@st.cache_resource
def load_llm():
    return pipeline("text2text-generation", model="t5-small")  # Substitua pelo Ollama ou outro modelo

# Processamento do arquivo
if uploaded_file:
    with st.spinner("Carregando o arquivo..."):
        if uploaded_file.type == "application/pdf":
            content = read_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            content = read_txt(uploaded_file)
        else:
            st.error("Formato de arquivo não suportado!")
            st.stop()

    # Exibir texto carregado
    st.subheader("Conteúdo do arquivo:")
    st.text_area("Texto carregado:", value=content[:5000], height=200)

    # Ação selecionada
    llm = load_llm()
    if st.button("Executar"):
        with st.spinner("Processando..."):
            if action == "Resumo":
                result = llm(content, max_length=100)
            elif action == "Extração de trecho":
                result = llm(f"Extraia trechos relevantes do seguinte texto:\n{content}", max_length=200)
            elif action == "Perguntas e respostas":
                question = st.text_input("Digite sua pergunta:")
                if question:
                    result = llm(f"Pergunta: {question}\nTexto: {content}", max_length=200)
                else:
                    st.warning("Digite uma pergunta para prosseguir!")
                    result = None

        # Exibir resultados
        if result:
            st.subheader("Resultado:")
            st.write(result)

            # Salvar resultado
            if save_download:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                    tmp_file.write(result.encode("utf-8"))
                    st.download_button(
                        "Baixar resultado",
                        data=result,
                        file_name="resultado.txt",
                        mime="text/plain"
                    )

# Logo e rodapé
st.markdown("---")
st.markdown("**Desenvolvido por Beatriz Cardoso Cunha**")


