---
theme: default
title: Data valuation and attribution
info: |
  ## A primer on data valuation and attribution
  Some examples of how to attribute data sources and how to value data in your projects
  using pyDVL.

  Learn more at [pydvl.org](https://pydvl.org)
class: text-center # apply any unocss classes to the current slide
highlighter: shiki # https://sli.dev/custom/highlighters.html
drawings:
  # https://sli.dev/guide/drawing
  persist: false
transition: slide-left # https://sli.dev/guide/animations#slide-transitions
mdc: true # https://sli.dev/guide/syntax#mdc-syntax
themeConfig:
  primary: '#084059'
hideInToc: true
---

# Data valuation and attribution

A mild introduction with practical applications


<div style="display: flex; justify-content: center;">
  <img class="w-40" src="/pydata-logo.png" alt="PyData logo" />
</div>


[Kristof Schröder](https://github.com/schroedk) - 
[Miguel de Benito Delgado](https://mdbenito.8027.org)


<div style="display: flex; justify-content: center; align-items:center;">
  <a href="https://transferlab.ai">
    <img class="w-40" src="/transferlab-logo.svg" alt="TransferLab logo" />
  </a>
</div>
<img class="abs-bl m-4 w-25"
     src="/institute-logo.svg"
     alt="appliedAI Institute logo" />

<div class="abs-br m-1 flex">
  <button @click="$slidev.nav.openInEditor()" title="Open in Editor" class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon:edit />
  </button>
  <a href="https://github.com/aai-institute/pydvl" target="_blank" alt="GitHub" title="Open in GitHub"
    class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon-logo-github />
  </a>
</div>

<!--
10 seconds intro, no personal details.
-->

---
transition: fade-out
layout: two-cols
hideInToc: true
---

## Plan for the talk

This is just temporary for ourselves

MAX OF 10 SLIDES!!!

::right::

<AutoFitText :max="26" :min="8">

<Toc v-click minDepth="1" maxDepth="1"></Toc>

</AutoFitText>

---
layout: fact
hideInToc: true
title: What
---

## What is data valuation?

---
title: What is data valuation?
level: 1
layout: two-cols
class: self-center text-center p-6
transition: fade-out
---

# Data valuation computes...


::right::

<v-click>

the **contribution** of a data point ...

</v-click>

<br>
<br>

<v-click>

...to the overall **model performance**

</v-click>

<br>

<v-click>

or

...to a **single prediction**

</v-click>


<!--

[click:2] What data valuation is not:
- Differences to SHAP, LIME, etc.
-->

---
layout: fact
title: Caveat
hideInToc: true
---

(Actually, it's a bit more complicated)

<!---
Intrinsic notions of value

Using different scorers,

etc.
-->

---
layout: fact
hideInToc: true
---

## What can data valuation do for you?

---
level: 1
title: "Example 1: Data cleaning"
layout: two-cols-header
class: p-6 table-center
---

## Example 1: Data cleaning


::left::

<v-clicks>

- [Top Hits Spotify from 2000-2019](https://www.kaggle.com/datasets/paradisejoy/top-hits-spotify-from-20002019)[^1]
- Predict song popularity
- Simple `GradientBoostingRegressor`
- Drop "bad" data points
</v-clicks>

<br>

<v-after>

| Data dropped | MAE improvement |
|--------------|-----------------|
| 10%          | 9%              |
| 15%          | 11%             |
| 20%          | 14%             |

</v-after>

::right::

<div v-click="'+1'">

### Three steps


</div>

```python {hide|1-2|4-5|7}
values = valuation.fit(data).values()
values.sort()

clean_data = data.drop_indices(values[:100].indices)
model.fit(clean_data)

assert model.score(test) > 1.05 * previous_accuracy
```
<br>

<div v-after class="text-center">

#### Profit!

TODO:  (of course not all that glitters is gold... etc. )
</div>

[^1]: https://www.kaggle.com/datasets/paradisejoy/top-hits-spotify-from-20002019

<!--
[click:5] Take these data with a pinch of salt...

[click:4] 1.05 is just a number for the slide of course
-->

---
title: Other tasks
level: 1
layout: two-cols-header
class: p-6
---

## What can data valuation do for you?

::left::

### This bears repeating:

```python
model.fit(clean_data)
assert model.score(test) > 1.05 * previous_accuracy
```

<v-clicks>

- Increase accuracy by removing bogus points
- Crucially, select data for manual inspection
- **Data debugging**<br>
  _what's wrong with this data?_
- **Model debugging**<br>
  _why are these data detrimental?_

</v-clicks>

::right::

<div v-click>

### But also:

- **Data acquisition**: prioritize data sources
- **Attribution**: find the most important data points

And more speculatively:

- **Continual learning**: compress your dataset
- **Data markets**: price your data
- Improve **fairness metrics**
- ...

</div>

---
layout: fact
hideInToc: true
---

## What do you need?

---
title: Requirements
level: 1
layout: two-cols
class: px-6 table-invisible
---

## Requirements


<v-clicks>

- Any scikit-learn model
- Or a wrapper with `fit()`, `predict()`, and `score()`
- An _imperfect_ dataset
- ```shell
  pip install pydvl
  ```
- (Some) elbow grease
- Compute

</v-clicks>


<v-click>

|   |  |  |
| --- | --- | --- |
| <a href="https://pydvl.org"> <img class="w-25" src="/pydvl-logo.svg" alt="pyDVL logo" /> </a> | + | <img class="w-45" src="/elbow-grease.png" alt="All-purpose remedy" /> |

</v-click>

::right::

<v-click>

## What frameworks?


- `numpy` and `sklearn`
- _Influence Functions_ use `pytorch`
- Planned: allow `jax` and `torch` everywhere
- `joblib` for parallelization with all of its backends
- `dask` for large models
- `memcached` for caching

</v-click>

<br><br><br>

<div v-after>

### pyDVL is still evolving!

</div>

---
layout: two-cols-header
class: pr-6 pt-6 table-center
---

## Example 2: Finding mislabeled data

::left::

<div v-click>

- Again: Predict song popularity
- Corrupt 5% of data at random setting<br>their popularity to 0
- Task: Detect these data points

</div>

<div v-click>

| % low values | Mislabeled data |
|--------------|-----------------|
| 10%          | 60%             |
| 15%          | 85%             |
| 20%          | 100%            |

</div>

::right::

<v-click>

### Three steps

````md magic-move

// First example
```python {none|1-2|3-4|5-9|all}
train, val, test = load_spotify_dataset(...)
model = GradientBoostingRegressor(n_estimators=10)
scorer = SupervisedScorer("accuracy", test)
utility = Utility(model, scorer)
valuation = DataShapleyValuation(
    utility, MSRSampler(), RankCorrelation()
)
with joblib.parallel_backend("loky", n_jobs=16):
    valuation.fit(train)
```

```python {2-3}
train, test = load_data()
model = AnyModel()
scorer = CustomScorer(test)
utility = Utility(model, scorer)
valuation = DataShapleyValuation(
    utility, MSRSampler(), RankCorrelation()
)
with joblib.parallel_backend("loky", n_jobs=16):
    valuation.fit(train)
```

```python {5-7}
train, test = load_data()
model = AnyModel()
scorer = CustomScorer(test)
utility = Utility(model, scorer)
valuation = AnyValuationMethod(
    utility, SomeSampler(), StoppingCriterion()
)
with joblib.parallel_backend("loky", n_jobs=16):
    valuation.fit(train)
```

```python {8-9}
train, test = load_data()
model = AnyModel()
scorer = CustomScorer(test)
utility = Utility(model, scorer)
valuation = AnyValuationMethod(
    utility, SomeSampler(), StoppingCriterion()
)
with joblib.parallel_backend("ray", n_jobs=48):
    valuation.fit(train)
```
````

</v-click>

<v-click>
<v-drag pos="813,188,80,80,36">
<div text-center>(New interface)</div>
</v-drag>
</v-click>

---
title: Measuring value with marginal contributions
level: 1
layout: two-cols-header
class: px-6
---

## One family of methods: marginal contributions

```python {1-2|3-5}
model = LogisticRegression()
train, test = Dataset.from_sklearn(load_iris(), train_size=0.6)
def u(data):
    model.fit(data)
    return model.score(test)
```

<div v-click class="text-center">

Take one data point $x$

</div>

::left::

<div v-click class="text-center">Take the whole dataset</div>

```python {hide|1|2|3|1-3}
score = u(train)
score_without = u(train.drop(x))
value = score - score_without
```

<div v-click class="text-center text-bold text-xl">Leave-One-Out</div>

::right::

<div v-click class="text-center">Look at subsets</div>

```python {hide|1-2|3|4|all}
for subset in sampler.from_data(train):
  scores.append[u(subset)]
  scores_without.append[u(subset.drop(x))]
value = weighted_mean(scores - scores_without, coefficients)
```
<div v-click="14" class="text-center text-bold text-xl">Semivalue (e.g. Shapley)</div>

---
layout: two-cols
title: Problems with data valuation
level: 1
---

## Where's the catch?

This is not a silver bullet

- <span v-mark.underline.red="3">Computational cost</span>
- <span v-mark.underline.red="3">Convergence</span>
- <span v-mark.strike-through.orange="2">Consistency</span>
- <span v-mark.underline.green="1">Model dependence</span>

TODO:
$O(2^n)$ ? But $O(n \log(n))$ for certain situations.

::right::

Janos:

- Only the metric used for data valuation is actually improved when dropping data; Other metrics even get worse. I tried MAE, MSE and MAPE
- The dataset is very small
- Detecting corrupted data only worked when using more max-draws and stricter convergence criteria than in the tutorial. The result is still dependent on the seed
- All of this is for one random seed!

<!--
[click] New methods are appearing that look at data distributions, independently of the model. Also, the model can be changed to a simpler one, or a surrogate model can be used (KNN-Shapley).

[click] One cannot look at value rankings, but must instead look at subsets of data. Also: focus on the extrema.

[click]

-->

---
layout: fact
title: Influence functions
level: 1
---

## The influence of a training point

(on single test points)


<!--

As opposed to the Shapley value, which looks at the average influence of a data point on the whole test set.

-->

---
title: The influence of a training point
level: 1
layout: two-cols-header
class: table-center p-6
---

## The influence of a training point

::left::

| Data                                  |                           | Test loss                |
|---------------------------------------|---------------------------|--------------------------|
| $\{z_1, z_2, ..., z_n\}$              | (... train ...) $\to$     | $L(z)$                   |
| $\{z_1, \red{\sout{z_2}}, ..., z_n\}$ | (... train ...) $\to$     | $L_{\red{-z_2}}(z)$      |


<v-click>

The "influence" of $z_i$ on test point $z$ is roughly

  $$L(z) - L_{-z_i}(z)$$


</v-click>

::right::

<v-clicks>

- One value per train / test point pair $(z_i, z)$

- A <span v-mark.underline.red="'4'">full retraining</span> per train / test point pair!

</v-clicks>

---
layout: two-cols-header
title: Computing influences
level: 1
class: p-6
---

## Computing influences

::left::


<span v-mark.underline.red>Luckily<span v-click="'1'">?</span></span>

$$
I(z_i, z) = \nabla_\theta L^\top \cdot H^{-1}_{\theta} \cdot \nabla_\theta L
$$

<v-click at="2">
<Arrow x1="360" y1="275" x2="325" y2="195" color="red"/>
</v-click>

::right::

<v-clicks>

- Inverse of the Hessian!
- Implicit Hessian-vector products
- Quality of the approximations
- Does it matter?

</v-clicks>

---
layout: two-cols-header
level: 1
class: table-invisible table-center py-6 no-bullet-points
---

## Example 3: Finding mislabeled cells


::left::

<v-clicks>

- [**NIH dataset**](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets) for malaria screening[^1]
- ![](/malaria/some-images-could-be-mislabelled.png)
- **Goal:** detect these data points with pyDVL

</v-clicks>

<div v-click>

| Uninfected | Infected |
|------------|----------|
| <img src="/malaria/C123P84ThinF_IMG_20151002_152144_cell_49.png" alt="Uninfected cell"> | <img src="/malaria/C39P4thinF_original_IMG_20150622_111206_cell_87.png" alt="Infected cell"> |

</div>

::right::
TODO: This after the results?
<br>
<v-click>

````md magic-move

```python {hide|4|5|7-8|10|all|4,7}
torch_model = ...  # Trained model
train, test = ... # Dataloaders

if_model = DirectInfluence(torch_model, loss, ...)
if_model.fit(train)

if_calc = SequentialInfluenceCalculator(if_model)
lazy_values = if_calc.influences(test, train)

values = lazy_values.to_zarr(path, ...)
```

```python {4,7}
torch_model = ...  # Trained model
train, test = ... # Dataloaders

if_model = ArnoldiInfluence(torch_model, loss, ...)
if_model.fit(train)

if_calc = SequentialInfluenceCalculator(if_model)
lazy_values = if_calc.influences(test, train)

values = lazy_values.to_zarr(path, ...)
```

```python {4,7}
torch_model = ...  # Trained model
train, test = ... # Dataloaders

if_model = NystroemSketchInfluence(torch_model, loss, ...)
if_model.fit(train)

if_calc = SequentialInfluenceCalculator(if_model)
lazy_values = if_calc.influences(test, train)

values = lazy_values.to_zarr(path, ...)
```

```python {4,7}
torch_model = ...  # Trained model
train, test = ... # Dataloaders

if_model = NystroemSketchInfluence(torch_model, loss, ...)
if_model.fit(train)

if_calc = DaskInfluenceCalculator(if_model)
lazy_values = if_calc.influences(test, train)

values = lazy_values.to_zarr(path, ...)
```
````

</v-click>

<div v-click="'+2'" class="text-center">

<br>

Plus CG, LiSSa, E-KFAC, ...

</div>


[^1]: https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria

---
hideInToc: true
layout: two-cols
class: p-6
---

## Results

<br>

<v-clicks>

- Compute all pairs of influences
- For each training point, compute the 25th percentile
- Sort training points by this value
- Look at the K smallest ones

</v-clicks>

::right::

<v-click>

Cells labelled as healthy
<img src="/malaria/smallest_3_0.25_quantile_influence_uninfected_uninfected.png" alt="Cells labelled as healthy" class="w-100">

</v-click>

<v-click>
<Arrow x1="650" y1="200" x2="585" y2="155" color="red"/>
<Arrow x1="655" y1="200" x2="705" y2="167" color="red"/>
</v-click>

<v-click>
Cells labelled as parasitized
<img src="/malaria/smallest_3_0.25_quantile_influence_parasitized_parasitized.png" alt="Cells labelled as parasitized" class="w-100">
</v-click>

<v-click>
<Arrow x1="600" y1="450" x2="670" y2="390" color="green"/>
</v-click>



---
layout: two-cols-header
---

## Accelerating IF computation


::left::
<v-clicks>

- Where the problem lies
- What fits in my GPU?
- What can we do

</v-clicks>

::right::

<v-clicks>

- Approximations
- Parallelization
- Out-of-core computation

</v-clicks>

::bottom::

```python
# Maybe an example here

```

---
title: Picking methods
level: 1
layout: two-cols-header
class: p-6 text-center no-bullet-points
---

## How to choose between IF and DV?

<br>

::left::

#### Influence functions

<v-clicks>
 
- Large models with costly retrainings
- `torch` interface
- Single test point

</v-clicks>

::right::

#### Data valuation

<v-clicks>

- Smaller models
- `sklearn` interface
- Value over the whole test set

</v-clicks>

::bottom::

<v-click>

### These are tools for <span v-mark.underline.orange>data debugging</span>!

</v-click>


---
layout: two-cols
title: Thank you!
hideInToc: true
class: text-center table-center table-invisible p-6
---
Thank you for your attention!
## [pydvl.org](https://pydvl.org)

<div style="display: flex; justify-content: center; align-items:center;">
  <a href="https://pydvl.org">
    <img class="w-25" src="/pydvl-logo.svg" alt="pyDVL logo" />
  </a>
</div>

::right::

PyDVL contributors


|  |  |  |
|--|--|--|
| [<img src="/people/anes.jpeg" alt="Anes Benmerzoug" class="author-thumbnail">](https://github.com/AnesBenmerzoug) | [<img src="/people/miguel.png" alt="Miguel de Benito Delgado" class="author-thumbnail">](https://github.com/mdbenito) | [<img src="/people/janos.jpeg" alt="Janoś Gabler" class="author-thumbnail">](https://github.com/janosg) | 
| [<img src="/people/jakob.jpeg" alt="Jakob Kruse" class="author-thumbnail">](https://github.com/jakobkruse1) | [<img src="/people/markus.jpeg" alt="Markus Semmler" class="author-thumbnail">](https://github.com/kosmitive) | [<img src="/people/fabio.png" alt="Fabio Peruzzo" class="author-thumbnail">](https://github.com/xuzzo) |
| [<img src="/people/kristof.jpg" alt="Kristof Schröder" class="author-thumbnail">](https://github.com/schroedk) | [<img src="/people/bastien.png" alt="Bastien Zim" class="author-thumbnail">](https://github.com/BastienZim) | [<img src="/people/uncle-sam.png" alt="You" class="author-thumbnail"><span style="font-size:small;">You!</span>](https://github.com/aai-institute/pydvl) |
