from DrissionPage import ChromiumPage, ChromiumOptions
import requests
import time

def coletar():
    co = ChromiumOptions().set_argument('--headless').set_argument('--no-sandbox').set_argument('--disable-gpu')
    # User-agent atualizado para 2026
    co.set_user_agent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36')
    
    page = ChromiumPage(co)
    webhook_url = "https://script.google.com/macros/s/AKfycbw2PsJB2nvpitaC3pA1Y0yX0r81YKbyYAMRWFrOuWiMu8xl7EFJz694nAazXz3CXVhq/exec"
    
    termos = ["impressão 3d articulado", "decoracao 3d", "3d culinaria", "suporte 3d"]
    todos_produtos = []

    # PASSO 1: Visita a Home para gerar cookies de "humano"
    print("Acessando home para validação...")
    page.get('https://shopee.com.br/')
    time.sleep(5)

    for termo in termos:
        print(f"Buscando: {termo}")
        # PASSO 2: Busca com delay
        page.get(f'https://shopee.com.br/search?keyword={termo}&sortBy=sales')
        
        # Simula rolagem humana para carregar o conteúdo
        for i in range(4):
            page.scroll.down(600)
            time.sleep(2)
        
        # PASSO 3: Tenta múltiplos seletores (caso a Shopee mude o código)
        itens = page.eles('xpath://div[@data-sqe="item"]')
        if not itens:
            itens = page.eles('.shopee-search-item-result__item') # Seletor secundário
            
        print(f"Encontrados {len(itens)} itens.")

        for item in itens[:10]:
            try:
                # Busca nome e preço de forma mais flexível
                nome = item.ele('xpath:.//div[contains(@class, "vvp_n")]').text or "Produto 3D"
                preco_raw = item.ele('xpath:.//div[contains(@class, "vvp_p")]').text or "0"
                preco = preco_raw.replace('R$', '').replace('.', '').replace(',', '.').strip()
                
                link = item.attr('href')
                link_completo = link if link.startswith('http') else 'https://shopee.com.br' + link

                todos_produtos.append({
                    "nome": nome,
                    "preco": preco,
                    "vendas": "Top Vendas",
                    "categoria": termo,
                    "link": link_completo
                })
            except:
                continue

    if todos_produtos:
        print(f"Sucesso! Enviando {len(todos_produtos)} itens...")
        requests.post(webhook_url, json=todos_produtos)
    else:
        # LOG DE DEPURAÇÃO: Se der 0, vamos ver o título da página que ele está vendo
        print(f"ALERTA: Nada encontrado. Título da página atual: {page.title}")
    
    page.quit()

if __name__ == "__main__":
    coletar()
