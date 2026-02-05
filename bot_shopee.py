from DrissionPage import ChromiumPage, ChromiumOptions
import requests
import time

def coletar_shopee():
    # Configuração para rodar no servidor do GitHub (sem interface visual)
    co = ChromiumOptions().set_argument('--headless').set_argument('--no-sandbox').set_argument('--disable-gpu')
    page = ChromiumPage(co)
    
    # URL do seu Google Apps Script
    webhook_url = "https://script.google.com/macros/s/AKfycbwLnBKOkxp8T5fgQ8DZkAOZZtMxf3BvXEyd-LdRwEPpxHxhNhBNVgDIEjhCfUnXUS0G/exec"
    
    # Termos de busca focados no seu negócio (Bambu Lab e Resina)
    termos = ["impressão 3d articulado", "decoracao 3d", "3d culinaria", "suporte 3d"]
    
    todos_produtos = []

    for termo in termos:
        print(f"Pesquisando: {termo}")
        url = f'https://shopee.com.br/search?keyword={termo}&sortBy=sales'
        page.get(url)
        time.sleep(5) # Espera carregar os produtos
        
        # O robô captura os itens da vitrine
        itens = page.eles('.shopee-search-item-result__item')
        
        for item in itens[:10]: # Pega os 10 mais vendidos de cada termo
            try:
                dados = {
                    "nome": item.ele('.vvp_n').text,
                    "preco": item.ele('.vvp_p').text.replace('R$', '').strip(),
                    "vendas_mensais": item.ele('.vvp_s').text,
                    "categoria": termo
                }
                todos_produtos.append(dados)
            except:
                continue # Pula se houver erro em algum item específico

    # Envia o relatório final para a sua planilha
    if todos_produtos:
        response = requests.post(webhook_url, json=todos_produtos)
        print(f"Relatório enviado! Status: {response.status_code}")
    
    page.quit()

if __name__ == "__main__":
    coletar_shopee()
