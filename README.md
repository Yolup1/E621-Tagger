
# E621 Tagger

A neural network for identifying tags in E621 posts.

### Requirement
* MatplotLib >= `3.24`
* Tensorflow >= `2.4`
* Pandas >= `1.2.3`
* Numpy >= `1.19.2`

### How it works

For the sake of being able to see the process of the architecture,<br>
the model is only saved using **weights**, which also means it is<br>
built during **runtime** and can be modified / loaded aftewards for<br>
use in other projects.<br>

##### Image Size

You **can** use any image size and disable the auto resizer (`512px`),<br>
this is not advised however as it was trained on `128px`, `256px` and `512px`.<br>


##### Location & Format

By default the model uses png files, if you want to change this,<br>
please edit `Tagger.py` and replace `source_dir` with the directory<br>
and file type of your desire.<br>
