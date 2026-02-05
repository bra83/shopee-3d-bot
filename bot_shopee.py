from DrissionPage import ChromiumPage, ChromiumOptions
import requests
import time

def coletar():
    # Configurações para não ser detectado como robô
    co = ChromiumOptions().set_argument('--headless').set_argument('--no-sandbox').set_argument('--disable-gpu')
    co.set_user_agent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36')
    
    page = ChromiumPage(co)
    
    # SEU LINK DO APPS SCRIPT (O mesmo que você usou no teste manual)
    webhook_url = "https://script.google.com/macros/s/AKfycbw2PsJB2nvpitaC3pA1Y0yX0r81YKbyYAMRWFrOuWiMu8xl7EFJz694nAazXz3CXVhq/exec"
    
    termos = ["impressão 3d articulado", "decoracao 3d", "3d culinaria", "suporte 3d"]
    todos_produtos = []

    for termo in termos:
        print(f"Buscando: {termo}")
        page.get(f'https://shopee.com.br/search?keyword={termo}&sortBy=sales')
        
        # Rola a página 3 vezes para carregar os itens (Shopee usa 'lazy load')
        for _ in range(3):
            page.scroll.down(1000)
            time.sleep(2)
        
        # Procura os cards de produtos
        itens = page.eles('xpath://div[@data-sqe="item"]')
        print(f"Encontrados {len(itens)} itens para {termo}")

        for item in itens[:10]: # Pega o Top 10
            try:
                nome = item.ele('.vvp_n').text if item.ele('.vvp_n') else "Produto 3D"
                # Limpa o preço para formato numérico
                preco_raw = item.ele('.vvp_p').text if item.ele('.vvp_p') else "0"
                preco = preco_raw.replace('R$', '').replace('.', '').replace(',', '.').strip()
                
                vendas = item.ele('.vvp_s').text if item.ele('.vvp_s') else "0"
                link = item.attr('href')
                link_completo = link if link.startswith('http') else 'https://shopee.com.br' + link

                todos_produtos.append({
                    "nome": nome,
                    "preco": preco,
                    "vendas": vendas,
                    "categoria": termo,
                    "link": link_completo
                })
            except:
                continue

    # Se encontrou algo, manda para a planilha
    if todos_produtos:
        print(f"Enviando {len(todos_produtos)} itens para o Google...")
        res = requests.post(webhook_url, json=todos_produtos)
        print("Resposta do Google:", res.text)
    else:
        print("ERRO: Nenhum produto encontrado. Verifique se o site da Shopee mudou.")

    page.quit()

if __name__ == "__main__":
    coletar()
