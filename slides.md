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

<br>

<div style="display: flex; justify-content: center;">
  <img class="w-40" src="/pydata-logo.png" alt="PyData logo" />
</div>

<br>
<br>

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

# Plan for the talk

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
- Predict song popularity with Gradient Boosting
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

<br>

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
class: px-6
---

## Requirements

<br>

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

<br>

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
<v-drag pos="660,330,80,80,-45">
<div text-center>(New interface)</div>
</v-drag>
</v-click>


---
title: Measuring value with marginal contributions
level: 1
layout: two-cols-header
class: px-6
---

## One family of methods

Marginal contributions

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
<br>
<div v-click class="text-center text-bold text-xl">Leave-One-Out</div>

::right::

<div v-click class="text-center">Look at subsets</div>

```python {hide|1-2|3|4|all}
for subset in sampler.from_data(train):
  scores.append[u(subset)]
  scores_without.append[u(subset.drop(x))]
value = weighted_mean(scores - scores_without, coefficients)
```
<br>
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

## Influence functions

---
title: The influence of a training point
level: 1
layout: center
---

## The influence of a training point

asdfas

<v-clicks>

- Major differences
- One value per train/test point pair
- Code?

</v-clicks>

---
layout: two-cols-header
---

## Example with IF

::left::

<v-clicks>

- Locating flipped labels in a dataset

</v-clicks>

::right::

```python {hide|1-2}
# Maybe an example here

```

---
layout: two-cols
---

# Example with IF (part 2)
<v-clicks>

- Data reweighthing, and how to use it

</v-clicks>


---
layout: two-cols-header
---

# Accelerating IF computation

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
---

## How to choose between IF and DV?

<v-clicks>

- 
- 

</v-clicks>





---
layout: two-cols
title: Thank you!
hideInToc: true
class: text-center table-center table-invisible p-6
---
Thank you for your attention!
## [pydvl.org](https://pydvl.org)

<br>
<div style="display: flex; justify-content: center; align-items:center;">
  <a href="https://pydvl.org">
    <img class="w-25" src="/pydvl-logo.svg" alt="pyDVL logo" />
  </a>
</div>

::right::

PyDVL contributors


|  |  |  |
|--|--|--|
| <img src="public/anes.jpeg" alt="Anes Benmerzoug" class="author-thumbnail"> | <img src="public/miguel.png" alt="Miguel de Benito Delgado" class="author-thumbnail"> | <img src="public/janos.jpeg" alt="Janoś Gabler" class="author-thumbnail"> | 
| <img src="public/jakob.jpeg" alt="Jakob Kruse" class="author-thumbnail"> | <img src="public/markus.jpeg" alt="Markus Semmler" class="author-thumbnail"> | <img src="public/fabio.png" alt="Fabio Peruzzo" class="author-thumbnail"> |
| <img src="public/kristof.png" alt="Kristof Schröder" class="author-thumbnail"> | <img src="public/bastian.png" alt="Bastian Zim" class="author-thumbnail"> | <img src="public/uncle-sam.png" alt="You" class="author-thumbnail"> |