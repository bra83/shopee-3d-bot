from DrissionPage import ChromiumPage, ChromiumOptions
import requests
import time

def coletar_shopee():
    # Configuração para rodar de forma invisível no GitHub
    co = ChromiumOptions().set_argument('--headless').set_argument('--no-sandbox').set_argument('--disable-gpu')
    page = ChromiumPage(co)
    
    # Seu endereço de entrega (Apps Script)
    webhook_url = "https://script.google.com/macros/s/AKfycbwDmkCFReAWkUFyLLj8I2Hj0TtCCFnCKvx7OCmMX79j3AQIIPEF8kbA7wHqvIYj5uk6/exec"
    
    # Seus termos de busca para o negócio 3D
    termos = ["impressão 3d articulado", "decoracao 3d", "3d culinaria", "suporte 3d"]
    
    todos_produtos = []

    for termo in termos:
        print(f"Buscando por: {termo}...")
        url = f'https://shopee.com.br/search?keyword={termo}&sortBy=sales'
        page.get(url)
        time.sleep(6) # Tempo de segurança para carregar a página
        
        # Localiza os itens na página
        itens = page.eles('.shopee-search-item-result__item')
        
        for item in itens[:10]: # Top 10 de cada categoria
            try:
                # Captura e limpa o preço para formato numérico
                preco_raw = item.ele('.vvp_p').text
                preco_limpo = preco_raw.replace('R$', '').replace('.', '').replace(',', '.').strip()

                # Captura o link do produto
                link = item.attr('href')
                link_completo = link if link.startswith('http') else 'https://shopee.com.br' + link

                dados = {
                    "nome": item.ele('.vvp_n').text,
                    "preco": preco_limpo,
                    "vendas_mensais": item.ele('.vvp_s').text,
                    "categoria": termo,
                    "link": link_completo
                }
                todos_produtos.append(dados)
            except Exception as e:
                continue

    # Envia os dados para a sua planilha
    if todos_produtos:
        try:
            res = requests.post(webhook_url, json=todos_produtos)
            print(f"Sucesso! {len(todos_produtos)} itens enviados. Status: {res.status_code}")
        except Exception as e:
            print(f"Erro ao enviar: {e}")
    
    page.quit()

if __name__ == "__main__":
    coletar_shopee()
