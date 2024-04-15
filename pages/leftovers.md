---
------
layout: two-cols-header
dragPos:
  square: 841,363,20,20,270
---

# Computing values with pyDVL

Three steps for all valuation methods

::left::

<v-clicks>

- <span v-mark.underline.orange="4">Prepare `Dataset` and `model`</span>
- <span v-mark.underline.green="5">Choose `Scorer` and `Utility`</span>
- <span v-mark.highlight.blue="6">Compute values (contribution to performance)</span>

</v-clicks>

<br>
<br>

<br>
<br>

::right::

```python {none|1-2|3-4|5-9|all}
train, test = Dataset.from_sklearn(load_iris(), train_size=0.6)
model = LogisticRegression()
scorer = SupervisedScorer("accuracy", test)
utility = Utility(model, scorer)
valuation = DataBanzhafValuation(
    utility, MSRSampler(), RankCorrelation()
)
with joblib.parallel_backend("ray", n_jobs=48):
    valuation.fit(train)
```


<div v-drag="'square'">
  <div class="i-material-symbols-check-circle-outline"></div>
</div>

<arrow v-click="[4,5]" x1="350" y1="260" x2="445" y2="245" color="orange" width="2" arrowSize="1" />
<arrow v-click="[5,6]" x1="350" y1="320" x2="445" y2="280" color="green" width="2" arrowSize="1" />


<style>
li {
  padding-top:2rem;
}
</style>

---
layout: 2x2-grid-header
level: 1
class: text-center
title: Measuring value
---

## Measuring relevance for model performance

Marginal-contribution-based methods

::topleft::

<div v-click="1" v-click.hide="+9">

Train on training data $D$
  
</div>

<div v-click="+9">

Train on **subsets** $S \subset D$
  
</div>

::topright::

<div v-click="2">

Compute accuracy on test data $T$

</div>

<div v-click="3" v-click.hide="+9">

$$ u(D)$$

</div>

<div v-click="+9">

$$ u(S) $$
  
</div>

::bottomleft::

<div v-click="4" v-click.hide="+9">

Train on training data $D_{-x}$

</div>

<div v-click="+9">

Train on **subsets** $S_{-x}$
  
</div>


::bottomright::

<div v-click="5">

Compute accuracy on test data $T$

</div>

<div v-click="6" v-click.hide="+9">

$$ u(D_{-x}) $$

</div>

<div v-click="+9">

$$ u(S_{-x}) $$
  
</div>

::bottom::

<v-drag pos="67,416,517,46"> <div text-center> <div v-click="7" v-click.hide="9" class="hide">

**Leave-One-Out:**$\quad \quad v(x) = u(D) - u(D_{-x})$

</div> </div> </v-drag>


<v-drag pos="424,433,517,66"> <div text-center> <div v-click="11">

**Shapley:** $\quad v(x) = \sum_{S} \  ... \ \left [ u(S \cup \{x\}) - u(S) \right ]$

</div> </div> </v-drag>


---
title: Measuring value with marginal contributions
level: 1
layout: two-cols-header
---

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

<v-clicks>

- Train and score
  ```python
  score = u(train)
   
  ```
- Train and score without $x$
  ```python
  score_without = u(train.drop(x))
  ```
- **Leave-One-Out** value:
  ```python
  value = score - score_without
  ```

</v-clicks>

<br>
<br>
<br>
<br>

::right::

<div v-click class="text-center">Look at subsets</div>

<v-clicks>

- Train and score
  ```python
  for sample in sampler.from_data(train):
    scores.append[u(sample)]
  ```
- Train and score without $x$
  ```python
    scores_without.append[u(sample.drop(x))]
  ```
- **(Shapley) Value**
  ```python
  value = weighted_mean(scores - scores_without, coefficients)
  ```

</v-clicks>

---
layout: image-right
image: data-valuation-taxonomy.svg
backgroundSize: contain
class: invertible
---

# Many methods for data valuation

It's a growing field [^1]

<v-clicks>

- Fit before, during, or after trainig
- With or without reference datasets
- Specific to classification / regression / unsupervised
- Different model assumptions (from none to strong)
- Local and global valuation

</v-clicks>

<!-- Footer -->

[^1]: [A taxonomy of data valuation](https://transferlab.ai/blog)


<!--
Notes can also sync with clicks

[click] pyDVL focuses around model-based, but we're introducing model-free methods as
well, e.g. LAVA.

[click] In some data market scenarios, one does not have a reference dataset, but
instead uses those available to construct one.

[click:3] Last click (skip two clicks)
-->
