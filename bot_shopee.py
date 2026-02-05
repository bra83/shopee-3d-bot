from DrissionPage import ChromiumPage, ChromiumOptions
import requests
import time

def coletar_shopee():
    co = ChromiumOptions().set_argument('--headless').set_argument('--no-sandbox').set_argument('--disable-gpu')
    page = ChromiumPage(co)
    
    webhook_url = "https://script.google.com/macros/s/AKfycbwDmkCFReAWkUFyLLj8I2Hj0TtCCFnCKvx7OCmMX79j3AQIIPEF8kbA7wHqvIYj5uk6/exec"
    termos = ["impressão 3d articulado", "decoracao 3d", "3d culinaria", "suporte 3d"]
    
    todos_produtos = []

    for termo in termos:
        print(f"--- Iniciando busca: {termo} ---")
        url = f'https://shopee.com.br/search?keyword={termo}&sortBy=sales'
        page.get(url)
        
        # Rola a página para baixo para carregar os produtos (Lazy Load)
        page.scroll.to_bottom()
        time.sleep(5) 
        
        # Tenta encontrar os produtos por um seletor mais genérico (data-sqe)
        itens = page.eles('xpath://div[@data-sqe="item"]')
        print(f"Encontrados {len(itens)} elementos para {termo}")
        
        for item in itens[:10]:
            try:
                # Busca o nome e preço dentro do item
                nome = item.ele('.vvp_n').text if item.ele('.vvp_n') else "Nome não encontrado"
                # Limpeza do preço
                preco_elem = item.ele('.vvp_p')
                preco = preco_elem.text.replace('R$', '').replace('.', '').replace(',', '.').strip() if preco_elem else "0"
                
                link = item.attr('href')
                link_completo = link if link.startswith('http') else 'https://shopee.com.br' + link

                todos_produtos.append({
                    "nome": nome,
                    "preco": preco,
                    "vendas_mensais": "Consultar", # Campo simplificado
                    "categoria": termo,
                    "link": link_completo
                })
            except Exception as e:
                print(f"Erro ao processar item: {e}")
                continue

    if todos_produtos:
        print(f"Enviando {len(todos_produtos)} produtos para a planilha...")
        res = requests.post(webhook_url, json=todos_produtos)
        print(f"Resposta do Google: {res.text}")
    else:
        print("ALERTA: Nenhum produto foi capturado. Verifique o site.")
    
    page.quit()

if __name__ == "__main__":
    coletar_shopee()
