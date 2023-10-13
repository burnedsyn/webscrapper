from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models.openai import ChatOpenAI
from datetime import datetime

def web_qa(url_list, query, out_name):
    openai = ChatOpenAI(
        model_name="gpt-4",
        max_tokens=5000
    )
    loader_list = []
    for i in url_list:
        print('loading url: %s' % i)
        loader_list.append(WebBaseLoader(i))

    index = VectorstoreIndexCreator().from_loaders(loader_list)
    ans = index.query(question=query,
                      llm=openai)
    print("")
    print(ans)

    outfile_name = out_name + datetime.now().strftime("%m-%d-%y-%H%M%S") + ".out"
    with open(outfile_name, 'w') as f:
        f.write(ans)

url_list = [
    "https://pandia.pro/guide/comment-utiliser-langchain-guide-pour-creer-un-bot-en-python/",
    "https://larevueia.fr/langchain-le-guide-essentiel/",
    "https://www.unite.ai/fr/zero-to-advanced-prompt-engineering-with-langchain-in-python/",
    "https://www.lemagit.fr/conseil/Lessentiel-sur-LangChain",
    "https://hackernoon.com/fr/bases-de-donn%C3%A9es-vectorielles-bases-de-la-recherche-vectorielle-et-du-package-langchain-en-python",
    "https://python.langchain.com/docs/get_started/introduction",
    "https://python.org",
]

prompt = '''
Vu le contexte, veuillez fournir ce qui suit :
1. Résumé de ce que c'est
2. Résumé de ce que cela fait
3. Résumé de comment l'utiliser
4. Fournir 10 idées de développement pour les débutants
'''

web_qa(url_list, prompt, "summary")