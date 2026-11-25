# EcoVision

Detecção, rastreamento e contagem de materiais recicláveis em tempo real com YOLO local
e estimativas de peso e valor por médias configuráveis.

## Interface web

Use Python 3.11 ou 3.12. Na raiz do projeto, crie o ambiente virtual, instale as dependências e abra a EcoVision:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-modern.txt
python main_modern.py
```

Para gravar a abertura, pressione **P** na tela inicial para entrar em tela cheia e
comece a gravação. Pressione **Enter** ou **Espaço** quando estiver pronto. A animação
começa 600 ms depois, e o painel aparece quando a animação e o aquecimento do modelo
terminarem. Clique em **Iniciar triagem** para abrir a câmera.

**P** e a opção **Modo apresentação** mantêm o painel completo em tela cheia.
Pressione **Esc** para voltar à janela. O menu também permite alternar o tema e
escolher a fonte da câmera enquanto a triagem estiver encerrada.
Use `--no-browser` para imprimir o endereço local e abri-lo manualmente.

Para uma demonstração controlada, pressione **1** para exibir e contabilizar as
detecções como garrafa, **2** para lata e **0** para voltar à classificação automática.
Os itens já contabilizados mantêm a categoria registrada no momento da contagem.

O símbolo do painel, o favicon e a tela de abertura usam o [SVG provisório](recicla/web/static/provisional-logo.svg) quando o arquivo opcional `ecovision.svg` está ausente da raiz.
Para animar suas partes separadamente, identifique o contorno fechado com
`id="ecovision-eye"` e a folha com `id="ecovision-leaf"`. Use `currentColor` nas
partes que devem acompanhar a cor da splash.

## Interface desktop anterior (macOS)

```bash
# 1) Python 3.11 ou 3.12 do python.org (ou: brew install python@3.12 python-tk@3.12)
cd EcoVision
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) descubra qual índice é a câmera do MacBook (seu código original usava 1; o teste_camera usava 0)
python tools/list_cameras.py --show
# ajuste em Configurações (dentro do app) ou em config/settings.yaml -> camera.source

# 3) rode
python main.py
```

Na primeira execução o macOS pede permissão de câmera para o Terminal/IDE
(Ajustes do Sistema › Privacidade e Segurança › Câmera). Sem internet, baixe antes `yolo11n.pt`
para `models/` (o Ultralytics o baixa sozinho quando há conexão).

## Como usar

| Controle | O que faz |
|---|---|
| **Iniciar / Pausar (Retomar) / Encerrar** | Liga a câmera e a IA; congela a imagem sem fechar a câmera; libera câmera e threads |
| **Nova sessão** | Salva a sessão atual em `sessions/` (CSV) e começa outra do zero |
| **Zerar** | Zera a contagem (pede confirmação; não salva) |
| **Exportar CSV** | `sessions/sessao_<data>_eventos.csv` e `_resumo.csv` |
| **Confiança mínima** | Deslizante em tempo real |
| **Configurações** | Câmera (índice ou **vídeo de demonstração**), resolução/FPS, pesos do modelo, dispositivo (auto/mps/cpu), `imgsz`, rastreamento, **peso médio e preço/kg por material** |

*Modo demonstração com vídeo gravado:* Configurações › Câmera › "Escolher vídeo…" (o vídeo repete em loop).

## Arquitetura

```
 thread câmera ──(só o último quadro)──▶ thread IA ──(FrameResult)──▶ thread principal (Tk)
 recicla/core/camera.py                 recicla/core/engine.py        recicla/ui/app.py
 abre/reconecta, resolução, FPS         YOLO + ByteTrack (detector.py) polling com after(): desenha caixas,
                                        regra de contagem (counting.py) HUD, estatísticas, histórico
                                        sessão/CSV (session.py)
```

* O Tkinter só é tocado na thread principal; as threads publicam dados (último resultado + fila de eventos).
* Memória constante: a câmera guarda 1 quadro; se a IA for mais lenta, quadros antigos são descartados.
* O modelo é carregado numa thread (a janela não trava) e reaproveitado entre Encerrar/Iniciar.
* Falhas tratadas: câmera sem permissão/índice errado, perda de sinal (reconecta), erro de inferência
  (sobrevive a falhas isoladas; 10 seguidas param com mensagem), MPS indisponível (cai para CPU).

```
recicla/core/   camera · detector (YOLO+tracker) · counting · estimates · session · engine · types
recicla/ui/     app (janela) · panels (estatísticas/histórico) · overlay (caixas/HUD) · settings_dialog · theme
config/         settings.yaml · materials.yaml (pesos/preços/cores) · bytetrack_recycle.yaml
training/       datasets.yaml · prepare_dataset.py · train.py · evaluate.py · export_model.py · README.md
tools/          list_cameras.py · capture_dataset.py · benchmark.py · yolo_webcam.py (seu original, preservado)
legacy/         seus arquivos originais, intactos (main_original.py, teste_tkinter_original.py, teste_camera_original.py)
tests/          testes automatizados (pytest) e scripts de verificação
```

## Como a contagem evita contar o mesmo objeto várias vezes

O ByteTrack dá um ID a cada objeto. Um ID só é **contado uma vez**, depois de `min_hits` quadros seguidos (padrão 5);
a categoria é decidida por **voto ponderado pela confiança** ao longo da trilha; e se o rastreador perder um objeto e
devolvê-lo com outro ID no mesmo lugar (mesma categoria, caixas sobrepostas, em até `reid_window_s` = 3 s), ele é
**reconhecido como o mesmo** e não é recontado. Na tela: caixa fina = candidato, caixa grossa com cantos = já contado.

Limites: objeto que sai de cena e volta depois da janela é contado de novo; um objeto *diferente* colocado
exatamente onde outro estava há < 3 s pode ser tratado como o mesmo (ajuste ou desligue `reid_window_s`).

## Peso e valor: são estimativas

Uma câmera RGB não mede peso. O sistema calcula `contagem × peso médio da categoria × preço/kg`, tudo configurável em
`config/materials.yaml` (ou na tela de Configurações). **Todos os números que vêm no arquivo são valores de exemplo**:
pese amostras reais do seu centro e use o preço da sua cooperativa. A interface e os CSVs rotulam tudo como estimativa.

## Modelo de detecção

A versão final usa um YOLO11n ajustado para as classes `garrafa` e `lata`. Solicite aos responsáveis pelo projeto uma cópia autorizada do checkpoint final e salve-a como `models/reciclagem_best.pt`. O arquivo original tem 5.513.873 bytes. O SHA-256 é `3cb5726f454980d7b1e543986a1ef5016077c920ecf134101599dcd09a5dfb11`.

Sem o checkpoint final, o aplicativo usa o YOLO11n genérico em modo demonstração. Nesse modo, o catálogo associa `bottle` a garrafa e `cup` a lata. Para uso offline, coloque os pesos genéricos oficiais em `models/yolo11n.pt` antes de executar.

O dispositivo `auto` seleciona MPS, CUDA ou CPU. Se a inferência via MPS falhar, o sistema usa CPU.

O [guia de treinamento](training/README.md) descreve como treinar pesos próprios. Para gerar um substituto de duas classes, use um dataset autorizado no formato YOLO com as classes `garrafa` e `lata` e passe seu manifesto ao script de treino com `--data`. Os dados e a receita exata do treino final não estão disponíveis aqui.

## Licenciamento

O código e os modelos YOLO da Ultralytics seguem os termos da AGPL-3.0 ou da licença Enterprise. O checkpoint final contém metadados do ambiente de treinamento e sua redistribuição depende da confirmação da procedência dos dados e dos direitos aplicáveis. A licença do EcoVision está pendente. [Termos da Ultralytics](https://www.ultralytics.com/license)

## Estado real do projeto

| Item | Estado |
|---|---|
| Captura, YOLO + ByteTrack, contagem única por objeto, sessão, estimativas, CSV, interface | **Implementado e testado** (ver abaixo) |
| Modelo treinado em recicláveis | Carregado de `models/reciclagem_best.pt` quando disponível |
| Precisão em recicláveis (mAP, confusões) | Avaliável com `training/evaluate.py` |
| Peso/preço | Valores de exemplo; precisam ser calibrados por você |
| Histórico de sessões | Grava `sessions/historico_sessoes.csv` (sem tela dedicada) |
| Gráficos | Barras de participação por categoria no painel (sem outros gráficos) |

**Limitações de fundo (não são bugs):**
* *Lata de alumínio × lata de aço* são visualmente quase idênticas; o modelo vai confundi-las. Considere fundir em
  uma classe "lata metálica" ou usar um sensor (ímã/indutivo) — o app não resolve isso só com a câmera.
* *Garrafa PET × plástica (PEAD)*: o critério visual (transparente × opaca) funciona só em parte.
* Datasets públicos são de outro "ambiente" (lixo na rua, fundo limpo); **fotos suas** (`tools/capture_dataset.py`) são o
  que mais melhora o resultado.

## Testes executados

No ambiente de desenvolvimento (**Linux, CPU, sem câmera física, sem macOS/MPS**):

* `pytest tests` — contagem/reidentificação, sessão, CSV, configuração, preparo de dataset, motor com vídeo e rastreador
  falso (threads, pausa, reinício, câmera inexistente, erros de inferência).
* `tests/ui_smoke.py` (sob display virtual) — a janela abre, inicia, pausa, abre Configurações e encerra; capturas inspecionadas.
* `tests/e2e_real_model.py` — YOLO11n real + ByteTrack + contagem + sessão com vídeo (IDs estáveis, `reset_tracker`, contagem única).
* `tests/smoke_training.py` — prepare → train → evaluate → export com dados **sintéticos** (prova que os scripts rodam;
  o modelo gerado não reconhece recicláveis).

**Não testado:** câmera real do MacBook, MPS, desempenho (FPS) no M1, aparência exata das fontes no macOS.

## Adicionar um material novo

1. `config/materials.yaml`: nova entrada (id, nome, cor, peso médio, preço, `model_classes`).
2. `training/datasets.yaml`: mapeie as classes de origem para os IDs do catálogo que você pretende treinar.
3. `python training/prepare_dataset.py && python training/train.py && python training/export_model.py`.

## Mapa dos seus arquivos originais

| Original | Destino | Motivo |
|---|---|---|
| `main.py` | `legacy/main_original.py`; o `main.py` atual é o novo ponto de entrada | A lógica (Tk + OpenCV + YOLO) foi mantida, mas a inferência rodava na thread da interface e travava a janela |
| `yolo_webcam.py` | `tools/yolo_webcam.py` (intacto) | Útil para testar modelo/câmera sem interface |
| `teste_camera.py` | `legacy/` + evoluído em `tools/list_cameras.py` | Só testava o índice 0 |
| `teste_tkinter.py` | `legacy/` | Esqueleto superado; tinha bug (`self.area_video` inexistente) |
