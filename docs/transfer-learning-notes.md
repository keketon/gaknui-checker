# 転移学習ノートブック 理解度メモ

`notebooks/training.ipynb`（MobileNetV3 Largeを使った転移学習）を自分の手で実装する過程で整理した、設計判断の理由や仕組みについてのQ&Aメモ。

## データ拡張 / 前処理

### Q. `train_transform`と`val_transform`で処理内容を分けているのはなぜ？

train用にのみ`RandomResizedCrop`や`RandomHorizontalFlip`のようなランダム性のある拡張をかけ、valには一切かけない。

- train側: データ拡張によって「同じ画像の少し違うバリエーション」を人工的に増やし、少ないデータ（191枚）でも過学習しにくくするため
- val側: 拡張をかけると、同じモデル・同じ重みで評価しても実行のたびにval_accがランダムにブレてしまう。エポック間でモデルが本当に改善しているかを正しく比較するには、**評価の条件を毎回固定して安定させる**必要がある

### Q. `Normalize`の`mean`/`std`はどこから来ている？自分のデータセットから計算した値を使わない理由は？

```python
weights = MobileNet_V3_Large_Weights.DEFAULT
preset = weights.transforms()
mean, std = preset.mean, preset.std
```

事前学習済みモデルが学習された際に使われたImageNetの統計値（`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`）を使っている。バックボーンの重みは「この分布で正規化された入力」を前提に学習されているため、自分のデータセット（kotone/saki画像）の統計値で正規化してしまうと、事前学習済みの特徴抽出層に想定と異なる分布のデータが入り、性能が落ちる。事前学習済みの重み（学習済みの値）は変更できないが、入力データ側をその分布に合わせることはできる。

## データセット / 層化分割

### Q. `dataset_train_view`と`dataset_val_view`という、同じディレクトリを指す`ImageFolder`を2つ作っているのはなぜ？

`ImageFolder`はインスタンスごとに1つの`transform`しか持てない仕様のため。train用とval用で異なるtransform（拡張の有無）を使い分けたいので、同じ`data_dir`を指す2つのインスタンスを作り、それぞれに別のtransformを渡している。実データの読み込み自体は`train_indices`/`val_indices`を`Subset`で適用することで振り分ける。

### Q. クラス（kotone/saki）ごとに分けてからシャッフル・分割する「層化分割」をしないと何が起こる？

全データを一括でシャッフルして先頭◯%をvalにすると、クラスごとの構成比が崩れる可能性がある。例えばsakiは90%がtrainに残るのにkotoneは70%しか残らない、といった偏りが起こり得る。クラスごとに分割してから合算することで、train/val両方でクラス比率を維持できる。

## モデル構築（転移学習）

### Q. バックボーン（`model.features`）と`classifier`の最初のLinear層を凍結（`requires_grad = False`）する理由は？

学習データが191枚と少なく、モデル全体（数百万パラメータ）を学習可能なままにすると簡単に過学習する。事前学習済みの特徴抽出能力を壊さずに保持しつつ、新しく追加した最終分類層のみを学習させることで、少ないデータでも安定して学習できる。

### Q. `requires_grad`はどこで決まる？

2段階で決まる。

1. **デフォルト**: `nn.Linear`や`nn.Conv2d`などの層が内部で持つ重みは`nn.Parameter`という型で、生成された瞬間に`requires_grad=True`がデフォルトで設定される。事前学習済みの値をロードしているかどうかとは独立した話。
2. **明示的な上書き**: `param.requires_grad = False`という属性代入によって個別に変更できる。

```python
def build_model(num_classes: int) -> nn.Module:
    model = mobilenet_v3_large(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier[0].parameters():
        param.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)  # 凍結ループより後に生成 → requires_grad=Trueのまま

    return model
```

最終Linear層の差し替えを凍結ループの**後**に書いているため、新しく作られたLinear層はどちらの凍結ループにも触れられず、デフォルトの`requires_grad=True`のまま残る。この順序が「バックボーンは凍結、最終層だけ学習可能」という状態を作っている。

### Q. `optimizer`に`model.parameters()`をそのまま渡さず`filter(lambda p: p.requires_grad, model.parameters())`を渡す理由は？

PyTorchのoptimizerは`step()`内で`p.grad is None`のパラメータを自動的にスキップする（内部状態の確保もこの後に行われるため、凍結パラメータに対して余計なメモリが確保されるわけでもない）ため、`filter`なしで全パラメータを渡しても実害はほぼない。それでも`filter`する理由は主に2つ。

- コードを読んだときに「今回学習対象なのはこのパラメータ群だけ」と意図が明確になる
- 将来的にバックボーンの一部を段階的に解凍してfine-tuningする場合、`filter`で絞った時点のoptimizerには後から解凍したパラメータが登録されないため、更新されない（optimizerの再構築や`add_param_group`が必要になる）。全パラメータを渡していればこの問題は起きない

## 学習ループ

### Q. 毎バッチ`optimizer.zero_grad()`を呼ぶ理由は？

PyTorchでは`.grad`はデフォルトで加算（累積）される仕様のため、リセットしないと前バッチまでの勾配が現在のバッチの勾配に混ざり続け、正しい学習ができなくなる。

### Q. `evaluate`で`torch.no_grad()`を使う理由は？結果の数値は変わる？

`no_grad()`はあくまで「backward用の計算グラフを構築するかどうか」の制御であり、forward計算自体の数値（loss・accuracy）には影響しない。目的はメモリと計算速度の節約で、評価時は`backward()`を呼ぶ予定がないため、逆伝播用に中間出力を保持する必要がない。

### Q. `best_val_acc`を使って「更新した時だけ」モデルを保存しているのはなぜ？

学習は必ずしも単調に改善するわけではなく、エポックを重ねるうちに過学習でval_accが下がることがある。無条件に毎エポック同じパスへ上書き保存すると、最終的に残るのは最後のエポックのモデルになってしまい、途中で最も良かった状態を失う。`best_val_acc`を追跡し、更新した瞬間だけ保存することで、学習終了後も常に「検証精度が最も良かった時点」の重みが残るようにしている。

### Q. 重み（`model_state_dict`）だけでなく`class_to_idx`も一緒に保存するのはなぜ？

モデルの出力はクラスのindex（0, 1, ...）でしかなく、そのindexが「kotone」なのか「saki」なのかという対応関係をモデル自身は持っていない。`ImageFolder`はディレクトリ名をアルファベット順にソートしてindexを割り当てる仕様のため、将来キャラクターを追加するなどでディレクトリ構成が変わると対応関係がズレる可能性がある。`class_to_idx`を重みと一緒に保存しておくことで、推論時にいつでも確実にindexからキャラクター名を復元できる。
