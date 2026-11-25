# Treinamento do modelo de recicláveis (100 % local)

A receita `datasets.yaml` usa a taxonomia de sete classes arquivada em `config/materials_7classes.yaml`. Ao usá-la, forneça ao preparo um diretório separado com esse catálogo salvo como `materials.yaml` por meio da opção `--materials-dir`. A configuração padrão da aplicação usa as classes `garrafa` e `lata`.

Para treinar um modelo compatível com a configuração padrão, use um dataset YOLO com essas duas classes e informe seu manifesto com `python training/train.py --data caminho/data.yaml`. Antes de exportar, preserve qualquer checkpoint final que já esteja em `models/reciclagem_best.pt`. Os passos abaixo descrevem o fluxo geral de treino e avaliação.

## 1. Datasets avaliados

| Dataset | Formato | O que cobre | Prós | Contras |
|---|---|---|---|---|
| **TACO** (Trash Annotations in Context) | COCO | 60 categorias de lixo, ~1.500 imagens; mapeadas para PET, plástica, lata de bebida (≈alumínio), lata de comida (≈aço), papelão, papel, embalagens | Cobre quase todas as suas classes; licença CC BY 4.0 nas anotações | Lixo "no mundo real" (rua/praia); poucas amostras de lata de aço e papel; só ~1.500 imagens |
| **Drinking Waste Classification** (Kaggle) | YOLO | Latas de alumínio, garrafas PET, frascos PEAD, garrafas de vidro | Imagens limpas de garrafas/latas, bom para reforçar essas classes | Fundo/estilo diferente do seu; precisa conta gratuita no Kaggle (ou baixar pelo navegador); confira os nomes das classes no seu download |
| **ZeroWaste-f** | COCO | Esteira de triagem real: papelão, plástico mole/rígido, metal | Domínio mais parecido com um centro de reciclagem | Só 4 classes grosseiras; desbalanceado (muito papelão, pouquíssimo metal) |
| **Suas fotos** (`tools/capture_dataset.py`) | YOLO | O que você quiser | Mesmo ambiente/câmera/luz do uso real: é o que mais melhora o resultado | Exige anotar (labelImg, CVAT ou Label Studio — gratuitos) |

Recomendação: **TACO + Drinking Waste (+ ZeroWaste) para a base e algumas centenas de fotos suas** para ajustar ao ambiente.
Combine tudo com `prepare_dataset.py`; ele unifica as classes e relata o que ficou fraco.

**Onde baixar** (confira os termos de cada um):
* TACO — https://github.com/pedropro/TACO (siga o README deles; coloque em `datasets/raw/taco/data/` com `annotations.json`)
* Drinking Waste Classification — procure o nome no Kaggle (`datasets/raw/drinking-waste/`)
* ZeroWaste — site oficial do projeto ZeroWaste (Boston University) (`datasets/raw/zerowaste/`)

## 2. Fluxo

```bash
# (a) ative as fontes que baixou em training/datasets.yaml (enabled: true) e confira os nomes das classes
python training/prepare_dataset.py --materials-dir caminho/catalogo-sete-classes

# (b) teste rápido (minutos) para ver se tudo roda e estimar o tempo por época
python training/train.py --epochs 1 --fraction 0.05

# (c) treino de verdade (MPS automático no M1; reduza --batch se faltar memória, --imgsz 512 acelera)
python training/train.py --epochs 100

# (d) avalie no conjunto de TESTE (precisão/recall/mAP por classe + matriz de confusão)
python training/evaluate.py

# (e) exporte para um caminho separado e preserve os pesos ativos
python training/export_model.py --dest models/reciclagem_novo.pt
```

Não medi o tempo de treino no M1; ele depende do tamanho do dataset. Use o passo (b) para estimar antes de começar.

## 3. Como interpretar o resultado

* Olhe **por classe**, não só o mAP geral: classes com poucas amostras (o `prepare_dataset.py` avisa) vão mal.
* Na matriz de confusão (`runs/eval/reciclagem/`), espere confusão **lata de alumínio × lata de aço** e **PET × plástica**:
  é limitação visual, não bug. Se atrapalhar, funda as classes ou use mais dados seus.
* O conjunto de teste só vale se as imagens de teste não forem quase duplicatas das de treino (fotos em sequência da mesma
  cena devem ficar juntas num mesmo conjunto: separe suas fotos por sessão/dia se for o caso).
* Só afirme "o modelo atinge X %" com o número que o `evaluate.py` imprimir no SEU treino.
