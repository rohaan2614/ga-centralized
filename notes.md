# Number of GPUs
* 1 GPU took: 1h 27min
* 2 GPU took: 20 min

| Job ID | Settings | Notes | 
|--------|----------|-------|
| 19194262 | LeNet5() - batch size 128 <br> no clipping <br> no ga | | 
| 19194283 | LeNet5() - batch size 128  <br> no clipping <br> ga | | 
| 19194285 | LeNet5() - batch size 128  <br> clipping <br> ga | | 
| 19194310 | ResNet18() - batch size 128  <br> no clipping <br> no ga | gres=gpu:4 | 
| 19194321 | ResNet18() - batch size 128  <br> no clipping <br> ga | gres=gpu:4  <br> cuda out of memory at 567/3911 | 
| 19194325 | ResNet18() - batch size 128  <br> clipping <br> ga | gres=gpu:4  <br> cuda out of memory at 567/3911 | 
| 19197460 | ResNet18() - **batch size 8**   <br> no clipping <br> ga | gres=gpu:4 | 
| 19197462 | ResNet18() - batch size 8 <br> **memory release in train()**   <br> no clipping <br> ga | gres=gpu:4  <br> cuda out of memory at 567/3911  | 
| 19197479 | ResNet18() - batch size 8 <br> memory release in train() **& GA()**   <br> no clipping <br> ga | gres=gpu:4  <br> GPU 0 has a total capacity of <u>79.10 GiB</u>| 
| 19197492 | ResNet18() - batch size 8 <br> memory release in train() & GA()   <br> no clipping <br> ga | **gres=gpu:2**  <br> GPU 0 has a total capacity of <u>39.38 GiB</u>| 
| 19197498 | ResNet18() - batch size 8 <br> memory release in train() & GA()   <br> no clipping <br> ga | **gres=gpu:1**  <br> GPU 0 has a total capacity of <u>39.38 GiB</u>| 
| 19197523 | ResNet18() - batch size 8 <br> memory release in train() & GA()   <br> no clipping <br> ga | **gres=h100:2**  <br> GPU 0 has a total capacity of <u>39.38 GiB</u>| 
| 19197524 | ResNet18() - batch size 8 <br> memory release in train() & GA()   <br> no clipping <br> ga | **gres=h100:4**  <br> GPU 0 has a total capacity of <u>39.38 GiB</u>| 
| 19197989 | LeNet5() - batch size 128 <br> no clipping <br> no ga | gres=gpu:2<br>**added L2 Norm Gradient to csv**<br>*=> These gradients can serve as a baseline to compare GA related dramatic losses* | 
| 19197993 | LeNet5() - batch size 128 <br> no clipping <br> **ga @ q=1e3** | gres=gpu:2 <br> *=> Accuracy <= 26%*| 
| 19198083 | **MNIST** LeNet5() - batch size 128 <br> no clipping <br>  no ga | gres=gpu:2 <br> *=> Gradients between 0 & 1* | 
| 19198087 | LeNet5() - batch size 64 <br> no clipping <br> **ga @ q=2e3** | gres=gpu:2 <br> *=> results stink* | 
| 19200212 | LeNet5() - batch size 128 <br> **clipping max = 4.0** <br> **ga @ q=1e3** | gres=gpu:2 <br> *=> results stink; accuracy falls to like 30%* |  
| 19199830 | **MNIST** LeNet5() - batch size 128 <br> no clipping <br>  ga @ q=10 | gres=gpu:2 *=> similar results as last time; L2 Norm Grad <= 0.8* |
| 19200225 | **CNN() (CIFAR-10)** - batch size 128 <br> no clipping <br>  no ga| gres=gpu:2 <br> *=> Model looks clean. Repeating exp with multiple qs. Confirming if a dramatic losses disappear in robust models* |
| 19202113 | CNN() (CIFAR-10) - batch size 128 <br> no clipping <br>  **ga @ q=1e0**  | gres=gpu:2 <br> *=>Not as accurate as without GA but reasonable*  |
| 19201652 | CNN() (CIFAR-10) - batch size 128 <br> no clipping <br>  **ga @ q=1e1**  | gres=gpu:2 <br> *Repeating exp: <br> model appears accurate and robust. No dramatic drops like before* |
| 19202115 | CNN() (CIFAR-10) - batch size 128 <br> no clipping <br>  **ga @ q=1e2**  | gres=gpu:2 <br> *=>Slightly more accurate than without GA* |

