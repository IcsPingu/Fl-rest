import re
import os

# --- CONFIGURAÇÃO ---
# Coloque aqui o nome EXATO dos seus arquivos de log
ARQUIVOS_DE_LOG = {
    "Cenário A (Pesos)": "resultado_cenario_A.txt",  # Ou o nome que você salvou
    "Cenário B (Features)": "resultado_cenario_B.txt"
}

def extrair_metricas(caminho_arquivo):
    """Lê o arquivo e retorna listas de rounds, acurácias e losses."""
    rounds = []
    acuracias = []
    losses = []
    
    if not os.path.exists(caminho_arquivo):
        print(f"❌ Erro: Arquivo '{caminho_arquivo}' não encontrado.")
        return None

    with open(caminho_arquivo, 'r', encoding='utf-8', errors='ignore') as f:
        conteudo = f.read()
        
        # Procura por linhas como: "--- Round 1 Acc: 25.50%, Loss: 2.1432 ---"
        # O regex captura: (Round), (Acurácia), (Loss)
        padrao = re.compile(r"Round\s+(\d+)\s+Acc:\s+([\d\.]+).*Loss:\s+([\d\.]+)")
        
        for match in padrao.finditer(conteudo):
            rounds.append(int(match.group(1)))
            acuracias.append(float(match.group(2)))
            losses.append(float(match.group(3)))
            
    return {"rounds": rounds, "acc": acuracias, "loss": losses}

def mostrar_comparacao():
    resultados = {}
    
    print("\n🔍 ANALISANDO RESULTADOS...")
    print("-" * 60)
    
    for nome_cenario, caminho in ARQUIVOS_DE_LOG.items():
        dados = extrair_metricas(caminho)
        if dados and dados['rounds']:
            # Pega os dados do último round registrado
            ult_acc = dados['acc'][-1]
            ult_loss = dados['loss'][-1]
            total_rounds = dados['rounds'][-1]
            
            resultados[nome_cenario] = {
                "acc": ult_acc,
                "loss": ult_loss,
                "rounds": total_rounds
            }
            print(f"✅ {nome_cenario}: Lido com sucesso ({total_rounds + 1} rounds).")
        else:
            print(f"⚠️ {nome_cenario}: Nenhum dado de treinamento encontrado no log.")

    # Tabela Final
    print("\n" + "="*60)
    print(f"{'CENÁRIO':<25} | {'ACURÁCIA FINAL':<15} | {'LOSS FINAL':<15}")
    print("="*60)
    
    melhor_acc = -1
    vencedor = None
    
    for nome, res in resultados.items():
        print(f"{nome:<25} | {res['acc']:.2f}%{'':<9} | {res['loss']:.4f}")
        
        if res['acc'] > melhor_acc:
            melhor_acc = res['acc']
            vencedor = nome
            
    print("-" * 60)
    
    if vencedor:
        diff = abs(resultados['Cenário A (Pesos)']['acc'] - resultados['Cenário B (Features)']['acc'])
        print(f"🏆 VENCEDOR: {vencedor}")
        print(f"📊 Diferença de Acurácia: {diff:.2f}%")
        
        if diff < 2.0:
            print("\n💡 CONCLUSÃO: Empate Técnico!")
            print("O Cenário B (Features) é preferível pois economiza comunicação")
            print("e permite modelos heterogêneos (conforme Qin et al.).")
        elif vencedor == "Cenário A (Pesos)":
            print("\n💡 CONCLUSÃO: O Cenário A venceu com margem significativa.")
            print("   Seus dados locais parecem ser complexos demais para serem")
            print("   aprendidos apenas via Logits (Features).")
    else:
        print("Não foi possível determinar um vencedor.")

if __name__ == "__main__":
    mostrar_comparacao()