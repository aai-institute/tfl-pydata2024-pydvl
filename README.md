# PyDVL @ PyData 2024 Berlin

These are the materials for the pyDVL presentation held on March 22nd, 2024 at
PyData Berlin.

## Detecting mislabelled images with pyDVL

In this example we use Influence Functions with pyDVL to detect mislabelled
images in a [**NIH
dataset**](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria)
with ~28K images for malaria screening.

To run the code, create a virtual environment and install the requirements:

```shell

python -m venv venv
source venv/bin/activate
pip install -r malaria/requirements.txt
```


## Slides

The slides use [Slidev](https://sli.dev/). To start the slideshow run:

```shell
cd slides
npm install
npm run dev
```

There's a live reload server running at http://localhost:3030

Edit the [slides.md](./slides/slides.md) to see the changes.


The slides are deployed with Netlify to
[pydvl-at-pydata2024.netlify.app](https://pydvl-at-pydata2024.netlify.app/)
upon every push to master.
