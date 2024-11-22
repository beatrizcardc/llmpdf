import streamlit as st
from transformers import pipeline
import tempfile
from PyPDF2 import PdfReader

# Configuração do Streamlit
st.set_page_config(page_title="Assistente PDF com IA", page_icon="📄", layout="wide")

# Função para carregar o modelo
@st.cache_resource
def load_model():
    return pipeline("summarization", model="t5-small", framework="tf")  # T5 Large para resumos mais completos

# Função para ler PDFs
def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Inicializa o app
st.title("Assistente PDF com IA 📄")

# Barra lateral
action = st.sidebar.selectbox("Escolha a ação", ["Resumo", "Perguntas e Respostas"])
save_download = st.sidebar.checkbox("Salvar resultado para download", value=True)

# Upload de arquivo
uploaded_file = st.file_uploader("Envie um arquivo PDF ou TXT", type=["pdf", "txt"])

if uploaded_file:
    # Processar o conteúdo do arquivo
    with st.spinner("Carregando o arquivo..."):
        if uploaded_file.type == "application/pdf":
            content = read_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            content = uploaded_file.read().decode("utf-8")
        else:
            st.error("Formato de arquivo não suportado!")
            st.stop()

    st.text_area("Texto carregado:", value=content[:5000], height=200)

    # Carregar modelo
    model = load_model()

    if action == "Resumo":
        if st.button("Gerar Resumo"):
            with st.spinner("Gerando resumo..."):
                # Gerar resumo do texto (de aproximadamente uma página)
                result = model(f"Resuma o seguinte texto:\n{content}", max_length=1000, min_length=300, do_sample=False)
                summary = result[0]["summary_text"] if result else "Nenhum resultado gerado."
                st.subheader("Resumo:")
                st.write(summary)

                # Salvar resultado
                if save_download:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                        tmp_file.write(summary.encode("utf-8"))
                        st.download_button(
                            "Baixar resumo",
                            data=summary,
                            file_name="resumo.txt",
                            mime="text/plain"
                        )

    elif action == "Perguntas e Respostas":
        question = st.text_input("Digite sua pergunta:")
        if st.button("Obter Resposta"):
            with st.spinner("Processando pergunta..."):
                if question.strip():
                    result = model(f"Pergunta: {question}\nTexto: {content}", max_length=500, do_sample=False)
                    answer = result[0]["summary_text"] if result else "Nenhuma resposta encontrada."
                    st.subheader("Resposta:")
                    st.write(answer)

                    # Salvar resultado
                    if save_download:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                            tmp_file.write(answer.encode("utf-8"))
                            st.download_button(
                                "Baixar resposta",
                                data=answer,
                                file_name="resposta.txt",
                                mime="text/plain"
                            )
                else:
                    st.warning("Por favor, digite uma pergunta válida.")


# Logo e rodapé
st.markdown("---")
st.markdown("**Desenvolvido por Beatriz Cardoso Cunha**")


