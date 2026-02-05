from DrissionPage import ChromiumPage, ChromiumOptions
import requests
import json

def coletar_shopee():
    # Configura o navegador para ser 'invisível'
    co = ChromiumOptions().set_argument('--headless').set_argument('--no-sandbox')
    page = ChromiumPage(co)
    
    # Busca por produtos 3D (Impressoras, Filamentos, etc)
    termo = "impressora 3d filamento"
    url = f'https://shopee.com.br/search?keyword={termo}&sortBy=sales'
    
    page.get(url)
    page.wait(5) # Espera a página carregar
    
    produtos = []
    # O robô olha os itens na tela
    itens = page.eles('.shopee-search-item-result__item')
    
    for item in itens[:20]: # Pega os 20 mais vendidos
        dados = {
            "nome": item.ele('.vvp_n').text if item.ele('.vvp_n') else "Sem nome",
            "preco": item.ele('.vvp_p').text if item.ele('.vvp_p') else "0",
            "vendas_mensais": item.ele('.vvp_s').text if item.ele('.vvp_s') else "0",
            "categoria": "3D Printing"
        }
        produtos.append(dados)
    
    # Envia o presente para o seu n8n
    webhook_url = "SUA_URL_DO_WEBHOOK_AQUI"
    requests.post(webhook_url, json=produtos)
    
    print(f"Enviado {len(produtos)} produtos para o n8n!")
    page.quit()

if __name__ == "__main__":
    coletar_shopee()
