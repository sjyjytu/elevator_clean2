# 复现命令

### 注意：

1. 如果训练设备不支持gpu，则把参数中--device后面的cuda:0改成cpu。
2. 如果训练设备的内存大小不支持，可以把参数中的--num-envs改小一点，这里默认是16，不支持的话可以改成8，4，2，1都可以，但是这个参数越小的话训练就会越慢。



### 上行高峰

训练命令

~~~
python smec_main.py --algo a2c --use-gae --lr 2e-5 --value-loss-coef 1 --num-envs 16 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 5 --real-data --exp-name smec_up_30_36 --eval-interval 50 --special-reward --num-env-steps 100000000 --device cuda:0 --test-num 10 --data-dir ./train_data/new/uppeak --dos 30:00-36:00 --use-attention
~~~

测试命令

~~~
python smec_main.py --evaluate --real-data --data-dir ./train_data/new/uppeak --exp-name smec_up_30_36 --seed 1 --dos 30:00-36:00
~~~



### 下行高峰

训练命令

~~~
python smec_main.py --algo a2c --use-gae --lr 2e-5 --value-loss-coef 1 --num-envs 16 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 5 --real-data --exp-name smec_dn_06_12 --eval-interval 50 --special-reward --num-env-steps 100000000 --device cuda:0 --test-num 10 --data-dir ./train_data/new/dnpeak --dos 06:00-12:00 --use-attention
~~~

测试命令

~~~
python smec_main.py --evaluate --real-data --data-dir ./train_data/new/dnpeak --exp-name smec_dn_06_12 --seed 1 --dos 06:00-12:00
~~~



### 午餐高峰

训练命令

~~~
python smec_main.py --algo a2c --use-gae --lr 2e-5 --value-loss-coef 1 --num-envs 16 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 5 --real-data --exp-name smec_lunch_00_06 --eval-interval 50 --special-reward --num-env-steps 100000000 --device cuda:0 --test-num 10 --data-dir ./train_data/new/lunchpeak --dos 00:00-06:00 --use-attention
~~~

测试命令

~~~
python smec_main.py --evaluate --real-data --data-dir ./train_data/new/lunchpeak --exp-name smec_lunch_00_06 --seed 1 --dos 00:00-06:00
~~~



### 非高峰

训练命令

~~~
python smec_main.py --algo a2c --use-gae --lr 2e-5 --value-loss-coef 1 --num-envs 16 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 5 --real-data --exp-name smec_not_00_06 --eval-interval 50 --special-reward --num-env-steps 100000000 --device cuda:0 --test-num 10 --data-dir ./train_data/new/notpeak --dos 00:00-06:00 --use-attention
~~~

测试命令

~~~
python smec_main.py --evaluate --real-data --data-dir ./train_data/new/notpeak --exp-name smec_not_00_06 --seed 1 --dos 00:00-06:00
~~~

