import streamlit as st
import os
from dotenv import load_dotenv
import nest_asyncio
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# --- CONFIGURAÇÃO INICIAL ---

# Carrega as variáveis de ambiente (nossa chave da API) do arquivo .env
nest_asyncio.apply()
load_dotenv()

# Configura o título e um ícone para a página da nossa ferramenta
st.set_page_config(page_title="Assistente de Manual", page_icon="🤖")

# Mostra o título principal na tela
st.title("🤖 Assistente de Manual do Sistema")
st.caption("Faça uma pergunta sobre o manual e a IA tentará responder.")

# --- FUNÇÕES DO BACKEND (O que acontece por trás dos panos) ---

# Usamos @st.cache_resource para que o carregamento e processamento do PDF 
# aconteça apenas uma vez, tornando a ferramenta mais rápida.
@st.cache_resource
def carregar_e_processar_documentos():
    """
    Esta função agora é mais inteligente. Ela verifica se o banco de dados vetorial
    já foi criado e salvo numa pasta. Se sim, apenas o carrega. Se não,
    cria o banco de dados e o salva para uso futuro.
    """
    # Define o nome da pasta onde o banco de dados será salvo
    pasta_db_vetorial = "banco_vetorial_chroma"

    # Modelo de embeddings que será usado
    modelo_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Verifica se a pasta do banco de dados já existe
    if os.path.exists(pasta_db_vetorial):
        # Se existe, apenas carrega o banco de dados que já foi processado
        st.info("Carregando banco de dados de vetores existente...")
        banco_vetores = Chroma(persist_directory=pasta_db_vetorial, embedding_function=modelo_embeddings)
        st.success("Banco de dados carregado com sucesso!")
    else:
        # Se não existe, faz o processo completo de carregar e processar
        st.info("Criando um novo banco de dados de vetores...")

        # 1. Carregar os documentos da pasta 'docs'
        carregador = PyPDFDirectoryLoader("docs")
        documentos = carregador.load()

        # 2. Dividir os documentos em pedaços menores
        divisor_texto = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        textos_divididos = divisor_texto.split_documents(documentos)

        # 3. Criar o banco de dados e salvá-lo na pasta definida
        banco_vetores = Chroma.from_documents(
            documents=textos_divididos,
            embedding=modelo_embeddings,
            persist_directory=pasta_db_vetorial # <--- Comando para salvar
        )
        st.success(f"Novo banco de dados criado e salvo com {len(documentos)} documento(s)!")

    return banco_vetores

# Tenta carregar os documentos. Se der erro na chave da API, mostra uma mensagem amigável.
try:
    banco_de_vetores = carregar_e_processar_documentos()

    # --- INTERFACE DO USUÁRIO ---

    # Cria um campo de texto para o usuário digitar a pergunta
    pergunta_usuario = st.text_input("Digite sua dúvida sobre o sistema:")

    # Se o usuário digitou algo, vamos processar a pergunta
    if pergunta_usuario:
        with st.spinner("Pensando..."): # Mostra uma animação de "carregando"
            
            # Configura o modelo de IA que vai gerar as respostas
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3, convert_system_message_to_human=True)
            
            # Cria a "cadeia" de busca e resposta (RetrievalQA)
            # É isso que conecta a pergunta do usuário, o banco de vetores e a IA.
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=banco_de_vetores.as_retriever(search_kwargs={"k": 3}), # Busca os 3 pedaços mais relevantes
                return_source_documents=True
            )

            # Executa a cadeia com a pergunta do usuário
            resultado = qa_chain({"query": pergunta_usuario})
            
            # Mostra a resposta na tela
            st.header("Resposta:")
            st.write(resultado["result"])

            # Opcional: Mostrar quais trechos do documento a IA usou para responder
            with st.expander("Ver fontes da resposta"):
                for doc in resultado["source_documents"]:
                    st.info(f"Fonte: {os.path.basename(doc.metadata.get('source', 'N/A'))}, Página: {doc.metadata.get('page', 'N/A')}")
                    st.markdown(f"> {doc.page_content[:250]}...") # Mostra um trecho do texto original

except Exception as e:
    st.error("Ocorreu um erro ao configurar a aplicação.")
    st.error(f"Detalhe do erro: {e}")
    st.warning("Verifique se sua chave GOOGLE_API_KEY está configurada corretamente no arquivo .env.")