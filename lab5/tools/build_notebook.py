"""Build lab5/sketch_lab.ipynb from the untouched starter notebook.

Fills in the TODO cells (Tasks 1-6), the Task 7 hooks (sketch augmentation and an auxiliary
classification loss), and a small local-machine patch (Apple-silicon MPS device, local paths,
saved Hugging Face token). Re-run this script after editing it; it never changes the starter.

    python tools/build_notebook.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / 'sketch_lab_starter.ipynb'
DST = HERE / 'sketch_lab.ipynb'

nb = json.loads(SRC.read_text())
cells = nb['cells']


def src(i):
    return ''.join(cells[i]['source'])


def set_src(i, text):
    cells[i]['source'] = text.splitlines(keepends=True)
    if cells[i]['cell_type'] == 'code':
        cells[i]['outputs'] = []
        cells[i]['execution_count'] = None


def replace_once(i, old, new):
    text = src(i)
    assert text.count(old) == 1, f'cell {i}: expected exactly one {old!r}'
    set_src(i, text.replace(old, new))


def find(pattern):
    hits = [i for i, c in enumerate(cells) if re.search(pattern, ''.join(c['source']))]
    assert len(hits) == 1, f'{pattern!r} matched cells {hits}'
    return hits[0]


def code_cell(text):
    return dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                source=text.splitlines(keepends=True))


def md_cell(text):
    return dict(cell_type='markdown', metadata={}, source=text.splitlines(keepends=True))


# Find every cell by its content first (indices shift once cells are inserted)
I_CFG = find(r'^CFG = dict\(')
I_SETUP = find(r'@title 2\. Setup')
I_AUG = find(r'^def sketch_transform')
I_HEAD = find(r'^class Head\(nn\.Module\)')
I_NCE = find(r'^class InfoNCE\(nn\.Module\)')
I_NCE_CHECK = find(r'@title Check Task 2')
I_BUILD = find(r'@title 12a\.')
I_TRAIN = find(r"TODO \(Task 3\)")
I_CURVES = find(r"TODO \(Task 4\)")
I_RETR = find(r"TODO \(Task 5\)")
I_CLASS = find(r"TODO \(Task 6\)")

# ---------------------------------------------------------------- Settings (section 1)
replace_once(I_CFG, "    TEMP_INIT=0.07,      # starting InfoNCE temperature (it is learned)\n",
             "    TEMP_INIT=0.07,      # starting InfoNCE temperature (it is learned)\n"
             "    LOSS='infonce',      # Task 7 B: 'infonce' or 'infonce_cls' (+ classification loss)\n"
             "    CLS_WEIGHT=0.5,      # Task 7 B: weight of the classification loss\n")
set_src(I_CFG, src(I_CFG).rstrip('\n') + '''

# ---- running on a local machine instead of Colab (ignored in Colab) ----
import sys
if 'google.colab' not in sys.modules:
    CFG.update(DATA_DIR='data', CACHE_DIR='data/cache', RUNS_DIR='runs',
               NUM_WORKERS=0)   # macOS cannot send notebook-defined Datasets to worker processes
''')

# ---------------------------------------------------------------- Setup (section 2): local patch
replace_once(I_SETUP, "import os, sys, csv, json, math, time, random, subprocess, zipfile\n",
             "import os, sys, csv, json, math, time, random, subprocess, zipfile\n"
             "os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')   # local patch: Apple GPU\n")
replace_once(I_SETUP, "HF_TOKEN = os.environ.get('HF_TOKEN')\n",
             "HF_TOKEN = os.environ.get('HF_TOKEN')\n"
             "if not HF_TOKEN and not IN_COLAB:   # local patch: the token saved by `hf auth login`\n"
             "    from huggingface_hub import get_token\n"
             "    HF_TOKEN = get_token()\n")
replace_once(I_SETUP,
             "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
             "USE_AMP = DEVICE == 'cuda'\n"
             "HARDWARE = torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'\n",
             "# Local patch: also use an Apple-silicon GPU (MPS); mixed precision stays CUDA-only (fp32 on MPS)\n"
             "if torch.cuda.is_available():\n"
             "    DEVICE = 'cuda'\n"
             "elif torch.backends.mps.is_available():\n"
             "    DEVICE = 'mps'\n"
             "else:\n"
             "    DEVICE = 'cpu'\n"
             "USE_AMP = DEVICE == 'cuda'\n"
             "if DEVICE == 'cuda':\n"
             "    HARDWARE = torch.cuda.get_device_name(0)\n"
             "elif DEVICE == 'mps':\n"
             "    chip = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],\n"
             "                          capture_output=True, text=True).stdout.strip()\n"
             "    HARDWARE = f'{chip} GPU (MPS)'\n"
             "else:\n"
             "    HARDWARE = 'CPU'\n")
replace_once(I_SETUP, "    torch.cuda.manual_seed_all(seed)\n",
             "    torch.cuda.manual_seed_all(seed)\n"
             "    if torch.backends.mps.is_available():   # local patch\n"
             "        torch.mps.manual_seed(seed)\n")

# ---------------------------------------------------------------- Section 9: sketch augmentation (Task 7 A)
set_src(I_AUG, '''def sketch_transform(train):
    """Sketch transform: augmented if training with AUG='strong'."""
    if train and CFG['AUG'] == 'strong':
        # Task 7 A: geometric changes only, so the strokes stay clean black lines on white.
        # People draw from memory, so a sketch is often shifted, scaled, tilted or mirrored
        # relative to its photo; these transforms teach the model to ignore exactly that.
        white = (255, 255, 255)
        aug = [
            T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.9, 1.1), fill=white),
        ]
        return T.Compose(aug + [T.ToTensor(), T.Normalize(IMNET_MEAN, IMNET_STD)])
    return base_tf


if CFG['AUG'] == 'strong':   # show what the augmentation does (figure for the report)
    _tf = sketch_transform(True)
    _sk = train_split.sketch_paths[0]
    fig, axes = plt.subplots(1, 6, figsize=(12, 2.3))
    axes[0].imshow(load_rgb(_sk).resize((IMG_SIZE, IMG_SIZE)))
    axes[0].set_title('original', fontsize=9)
    for ax in axes[1:]:
        x = _tf(load_rgb(_sk))
        x = x * torch.tensor(IMNET_STD)[:, None, None] + torch.tensor(IMNET_MEAN)[:, None, None]
        ax.imshow(x.permute(1, 2, 0).clamp(0, 1))
        ax.set_title('augmented', fontsize=9)
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(RUN_DIR / 'aug_examples.png', dpi=120)
    plt.show()
''')

# ---------------------------------------------------------------- Task 1: projection head
set_src(I_HEAD, '''class Head(nn.Module):
    """Projection head: Linear -> GELU -> Linear."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        """[B, in_dim] -> [B, out_dim]."""
        return self.net(x)
''')

# ---------------------------------------------------------------- Task 2: InfoNCE
replace_once(I_NCE,
             "        # TODO: steps 1-5 in the markdown cell above\n"
             "        raise NotImplementedError('TODO (Task 2): InfoNCE.forward')",
             "        # 1. L2-normalise (in fp32: under amp() the model output is 16-bit)\n"
             "        s = F.normalize(s.float(), dim=-1)\n"
             "        p = F.normalize(p.float(), dim=-1)\n"
             "        # 2. B x B cosine similarities, divided by the temperature\n"
             "        logits = (s @ p.T) * self.scale()\n"
             "        # 3-4. Row i should pick column i (sketch -> photo) and column i should pick row i\n"
             "        target = torch.arange(len(s), device=s.device)\n"
             "        loss_s2p = F.cross_entropy(logits, target)\n"
             "        loss_p2s = F.cross_entropy(logits.T, target)\n"
             "        # 5. Symmetric average\n"
             "        return 0.5 * (loss_s2p + loss_p2s)\n")

# Task 7 B: auxiliary classification loss (new cell after the Task 2 check)
cls_cell = code_cell('''# Task 7 B: InfoNCE + an auxiliary classification loss (used only when CFG['LOSS'] == 'infonce_cls')
class InfoNCECls(InfoNCE):
    """InfoNCE plus cross-entropy over the 125 classes, on the sketch embedding.

    The Sketchy paper found that adding a classification loss to its triplet loss raised R@1
    from 22.8% to 37.1%. Here a linear classifier on the sketch embedding s predicts the class of
    the photo it was drawn from (pidx gives the photo, so its class).
    """

    def __init__(self, init_temp, photo_cls, n_classes, dim, weight=0.5):
        super().__init__(init_temp)
        self.classifier = nn.Linear(dim, n_classes)
        self.weight = weight
        self.register_buffer('photo_cls', photo_cls.clone(), persistent=False)   # photo -> class

    def forward(self, s, p, pidx):
        contrastive = super().forward(s, p, pidx)
        labels = self.photo_cls[pidx]
        classification = F.cross_entropy(self.classifier(s.float()), labels)
        return contrastive + self.weight * classification


# Quick self-test on random data (no GPU needed)
_lf = InfoNCECls(0.07, torch.arange(8) % 4, 4, FEAT_DIM)
_l = _lf(torch.randn(8, FEAT_DIM), torch.randn(8, FEAT_DIM), torch.arange(8))
assert _l.dim() == 0 and math.isfinite(_l.item())
print(f'InfoNCECls ready (random-data loss {_l.item():.2f}).')
''')
cls_md = md_cell('''### Task 7 B: auxiliary classification loss
Used only for the Task 7 runs with `LOSS='infonce_cls'`. The base run uses plain InfoNCE.''')
cells[I_NCE_CHECK + 1:I_NCE_CHECK + 1] = [cls_md, cls_cell]
SHIFT = 2   # every index after the Task 2 check moves by two

# ---------------------------------------------------------------- 12a: pick the loss (Task 7 hook)
I_BUILD += SHIFT
replace_once(I_BUILD, "loss_fn = InfoNCE(CFG['TEMP_INIT']).to(DEVICE)\n",
             "if CFG.get('LOSS', 'infonce') == 'infonce_cls':   # Task 7 B hook\n"
             "    loss_fn = InfoNCECls(CFG['TEMP_INIT'], train_split.photo_cls, len(CLASSES), FEAT_DIM,\n"
             "                         CFG['CLS_WEIGHT']).to(DEVICE)\n"
             "else:\n"
             "    loss_fn = InfoNCE(CFG['TEMP_INIT']).to(DEVICE)\n")

# ---------------------------------------------------------------- Task 3: training loop
I_TRAIN += SHIFT
set_src(I_TRAIN, '''# ---- 3a. Optimiser and learning-rate schedule ----
# The temperature gets its own group with no weight decay (decay would pull log(1/tau) towards 0).
# Any other loss parameters (the Task 7 B classifier) train like the head.
temp_params = [loss_fn.log_scale]
other_loss_params = [p for name, p in loss_fn.named_parameters() if name != 'log_scale']
param_groups = [
    {'params': list(sketch_model.head.parameters()) + other_loss_params, 'lr': CFG['LR_HEAD']},
    {'params': temp_params, 'lr': CFG['LR_HEAD'], 'weight_decay': 0.0},
]
if FINETUNE:
    dino_params = [p for p in sketch_backbone.parameters() if p.requires_grad]
    param_groups.append({'params': dino_params, 'lr': CFG['LR_BACKBONE']})
opt = torch.optim.AdamW(param_groups, weight_decay=CFG['WEIGHT_DECAY'])

total_steps = EPOCHS * n_train_batches()
warmup_steps = max(1, min(CFG['WARMUP_STEPS'], int(0.1 * total_steps)))


def lr_factor(step):
    """Multiplier on each group's learning rate: linear warm-up from 0, then cosine to 0."""
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
print(f'{total_steps} steps: {warmup_steps} warm-up, then cosine decay to 0')

scaler = torch.amp.GradScaler(DEVICE, enabled=USE_AMP)   # for mixed precision (given)
best_score = -1.0
t_start = time.time()

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # ---- 3b. One epoch of training ----
    sketch_model.train()   # embed_split() switched it to eval mode last epoch
    loss_sum, n_batches = 0.0, 0
    for x, pidx in train_batches():
        opt.zero_grad(set_to_none=True)
        with amp():
            s = sketch_model(x)
        p = train_split.photo_feats[pidx]
        loss = loss_fn(s, p, pidx)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        loss_sum += loss.item()
        n_batches += 1
    train_loss = loss_sum / n_batches

    # ---- 3c. Validation ----
    # The validation loss is plain InfoNCE for every run, so runs with extra loss terms compare fairly
    sketch_model.eval()
    val_sum, val_n = 0.0, 0
    with torch.no_grad():
        for x, pidx in val_loss_batches():
            with amp():
                s = sketch_model(x)
            p = val_split.photo_feats[pidx]
            val_sum += InfoNCE.forward(loss_fn, s, p, pidx).item() * len(pidx)
            val_n += len(pidx)
    val_loss = val_sum / val_n
    vr = evaluate(*embed_split(sketch_model, val_split), val_split)
    log_epoch(epoch, train_loss, val_loss, vr, t0)   # given: saves to history and prints

    # ---- 3d. Keep the best model ----
    if vr['s2p_inst'][1] > best_score:
        best_score = vr['s2p_inst'][1]
        best_epoch = epoch
        save_best()

train_minutes = (time.time() - t_start) / 60
print(f'Best validation instance R@1: {best_score:.1f}% at epoch {best_epoch}; '
      f'{train_minutes:.1f} min; final temperature {loss_fn.temperature():.4f}')
''')

# ---------------------------------------------------------------- Task 4: curves
I_CURVES += SHIFT
set_src(I_CURVES, '''# Task 4: training curves
epochs = [h['epoch'] for h in history]
best_h = max(history, key=lambda h: h['val_inst_r1'])

fig, (ax_loss, ax_r1) = plt.subplots(1, 2, figsize=(12, 4.2))
ax_loss.plot(epochs, [h['train_loss'] for h in history], 'o-', ms=3, label='training loss')
ax_loss.plot(epochs, [h['val_loss'] for h in history], 's-', ms=3, label='validation loss (InfoNCE)')
ax_loss.set_xlabel('epoch')
ax_loss.set_ylabel('loss')
ax_loss.set_title('Loss')

ax_r1.plot(epochs, [h['val_inst_r1'] for h in history], 'o-', ms=3, label='instance (exact photo)')
ax_r1.plot(epochs, [h['val_cat_r1'] for h in history], 's-', ms=3, label='category (right class)')
ax_r1.axvline(best_h['epoch'], color='grey', ls='--', lw=1)
ax_r1.annotate(f"best {best_h['val_inst_r1']:.1f}% (epoch {best_h['epoch']})",
               (best_h['epoch'], best_h['val_inst_r1']), textcoords='offset points',
               xytext=(-10, -18), ha='right', fontsize=9)
ax_r1.set_xlabel('epoch')
ax_r1.set_ylabel('validation sketch -> photo R@1 (%)')
ax_r1.set_title('Validation R@1')

for ax in (ax_loss, ax_r1):
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_xticks(epochs if len(epochs) <= 20 else epochs[::5])
fig.suptitle(f'{RUN_NAME}  ({HARDWARE})', fontsize=10)
plt.tight_layout()
plt.savefig(RUN_DIR / 'curves.png', dpi=150)
plt.show()

# Extra (for Q2): how the learned temperature moved
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot([0] + epochs, [CFG['TEMP_INIT']] + [h['temperature'] for h in history], 'o-', ms=3)
ax.set_xlabel('epoch')
ax.set_ylabel('temperature τ')
ax.set_title('Learned InfoNCE temperature')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RUN_DIR / 'temperature.png', dpi=150)
plt.show()
print(f"temperature: {CFG['TEMP_INIT']} -> {history[-1]['temperature']:.4f} "
      f"(lowest allowed: 0.01)")
''')

# ---------------------------------------------------------------- Task 5: retrieval examples
I_RETR += SHIFT
set_src(I_RETR, '''# Task 5: retrieval examples in both directions
sim = test_sk.float() @ test_ph.float().T           # [sketches, photos] cosine similarities
s_cls, s_pidx, p_cls = test_split.sketch_cls, test_split.sketch_pidx, test_split.photo_cls

# Rank of the exact photo for every sketch (0 = first)
exact_score = sim[torch.arange(len(sim)), s_pidx]
exact_rank = (sim > exact_score[:, None]).sum(1)
top1_photo = sim.argmax(1)


def pick(candidates, n, rng, used_classes):
    """Up to n candidates from different classes (deterministic, seeded)."""
    candidates = list(candidates)
    rng.shuffle(candidates)
    chosen = []
    for c in candidates:
        cls = int(s_cls[c]) if c < len(s_cls) else None
        if cls in used_classes:
            continue
        chosen.append(c)
        used_classes.add(cls)
        if len(chosen) == n:
            break
    return chosen


def show_image(ax, path, title, color=None, lw=4):
    ax.imshow(load_rgb(path).resize((224, 224)))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color or 'black')
        spine.set_linewidth(lw if color else 1)


GREEN, RED = '#2ca02c', '#d62728'
rng = random.Random(0)
used = set()
successes = pick(torch.nonzero(exact_rank == 0).flatten().tolist(), 3, rng, used)   # exact photo first
wrong_class = torch.nonzero(p_cls[top1_photo] != s_cls).flatten().tolist()
failures = pick(wrong_class, 2, rng, used)                                           # top-1 is the wrong class
s2p_queries = successes + failures

# 1) Sketch -> photo
fig, axes = plt.subplots(len(s2p_queries), 6, figsize=(12, 2.25 * len(s2p_queries)))
for row, j in enumerate(s2p_queries):
    show_image(axes[row, 0], test_split.sketch_paths[j], f'query: {CLASS_NAMES[s_cls[j]]}')
    for k, i in enumerate(sim[j].topk(5).indices.tolist()):
        right_class = int(p_cls[i]) == int(s_cls[j])
        title = CLASS_NAMES[p_cls[i]] + ('  ★ exact' if i == int(s_pidx[j]) else '')
        show_image(axes[row, k + 1], test_split.photo_paths[i], title, GREEN if right_class else RED)
    axes[row, 0].set_ylabel(f'exact at\\nrank {int(exact_rank[j]) + 1}', fontsize=8)
fig.suptitle('Sketch → photo: top-5 test photos (green = right class, red = wrong; ★ = the exact photo)',
             fontsize=10)
plt.tight_layout()
plt.savefig(RUN_DIR / 'retrieval_s2p.png', dpi=130)
plt.show()

# 2) Photo -> sketch (only photos that have test sketches)
sim_p2s = sim.T                                       # [photos, sketches]
photos = test_split.photos_with_sketches
own_top1 = [i for i in photos if int(s_pidx[sim_p2s[i].argmax()]) == i]
not_own = [i for i in photos if int(s_pidx[sim_p2s[i].argmax()]) != i]
rng = random.Random(1)
used_p = set()


def pick_photos(cands, n):
    cands = list(cands)
    rng.shuffle(cands)
    out = []
    for i in cands:
        if int(p_cls[i]) not in used_p:
            out.append(i)
            used_p.add(int(p_cls[i]))
        if len(out) == n:
            break
    return out


p2s_queries = pick_photos(own_top1, 2) + pick_photos(not_own, 2)
fig, axes = plt.subplots(len(p2s_queries), 6, figsize=(12, 2.25 * len(p2s_queries)))
for row, i in enumerate(p2s_queries):
    show_image(axes[row, 0], test_split.photo_paths[i], f'query: {CLASS_NAMES[p_cls[i]]}')
    for k, j in enumerate(sim_p2s[i].topk(5).indices.tolist()):
        right_class = int(s_cls[j]) == int(p_cls[i])
        title = CLASS_NAMES[s_cls[j]] + ('  ★ exact' if int(s_pidx[j]) == i else '')
        show_image(axes[row, k + 1], test_split.sketch_paths[j], title, GREEN if right_class else RED)
    n_own = len(test_split.sketches_of[i])
    axes[row, 0].set_ylabel(f'{n_own} sketches\\nof this photo', fontsize=8)
fig.suptitle('Photo → sketch: top-5 test sketches (green = right class, red = wrong; ★ = drawn from this photo)',
             fontsize=10)
plt.tight_layout()
plt.savefig(RUN_DIR / 'retrieval_p2s.png', dpi=130)
plt.show()

# Numbers for Q8
print('Sketch -> photo queries (row: class, rank of exact photo, top-1 class):')
for j in s2p_queries:
    print(f'  {CLASS_NAMES[s_cls[j]]:>18s}: exact at rank {int(exact_rank[j]) + 1:4d}, '
          f'top-1 = {CLASS_NAMES[p_cls[top1_photo[j]]]}')
''')

# ---------------------------------------------------------------- Task 6: prototypes
I_CLASS += SHIFT
set_src(I_CLASS, '''# Task 6: classify each test sketch with class prototypes
# 1. Prototypes: normalise each training photo feature, average per class, normalise again
photo_f = F.normalize(train_split.photo_feats.float().cpu(), dim=-1)
n_classes = len(CLASSES)
protos = torch.zeros(n_classes, photo_f.shape[1]).index_add_(0, train_split.photo_cls, photo_f)
protos = F.normalize(protos, dim=-1)                       # [125, 768]

# 2. Guess: the most similar prototype (test_sk is already normalised)
guess = (test_sk.float().cpu() @ protos.T).argmax(1)
truth = test_split.sketch_cls

# 3. Accuracy over all test sketches
class_acc = (guess == truth).float().mean().item() * 100
print(f'Class-from-list accuracy: {class_acc:.1f}% of {len(truth):,} test sketches '
      f'(random guessing: {100 / n_classes:.1f}%)')
print(f"For comparison, category R@1 (nearest test photo's class): {test_results['s2p_cat'][1]:.1f}%")

# Extra (for Q7): the most common confusions
pairs = {}
for t, g in zip(truth.tolist(), guess.tolist()):
    if t != g:
        pairs[(t, g)] = pairs.get((t, g), 0) + 1
print('Most common mistakes (true -> guessed: count):')
for (t, g), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:6]:
    print(f'  {CLASS_NAMES[t]} -> {CLASS_NAMES[g]}: {n}')

# 4. Figure: 8 test sketches (5 right, 3 wrong when possible), green = right, red = wrong
rng = random.Random(0)
right = [j for j in range(len(truth)) if guess[j] == truth[j]]
wrong = [j for j in range(len(truth)) if guess[j] != truth[j]]
n_wrong = min(3, len(wrong))
shown = rng.sample(right, 8 - n_wrong) + rng.sample(wrong, n_wrong)
fig, axes = plt.subplots(2, 4, figsize=(10, 6.6))
for ax, j in zip(axes.flat, shown):
    ok = guess[j] == truth[j]
    ax.imshow(load_rgb(test_split.sketch_paths[j]).resize((224, 224)))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'guess: {CLASS_NAMES[guess[j]]}\\ntrue: {CLASS_NAMES[truth[j]]}', fontsize=9,
                 color='#2ca02c' if ok else '#d62728')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2ca02c' if ok else '#d62728')
        spine.set_linewidth(4)
fig.suptitle(f'Class from a list of {n_classes}: accuracy {class_acc:.1f}%', fontsize=11)
plt.tight_layout(h_pad=2.5)
plt.savefig(RUN_DIR / 'class_guesses.png', dpi=130)
plt.show()
''')

DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
print('wrote', DST)

# ---------------------------------------------------------------- Submission cell: the base run's folder
nb = json.loads(DST.read_text())
i_zip = next(i for i, c in enumerate(nb['cells']) if "BASE_RUN = ''" in ''.join(c['source']))
text = ''.join(nb['cells'][i_zip]['source']).replace(
    "BASE_RUN = ''", "BASE_RUN = 'dinov3-vits16-pretrain-lvd1689m_blocks12_aug-none'", 1)
nb['cells'][i_zip]['source'] = text.splitlines(keepends=True)
DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
print('BASE_RUN set')
