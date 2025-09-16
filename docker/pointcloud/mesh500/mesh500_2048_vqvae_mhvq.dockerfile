FROM kohido/base_dl_cuda129:v0.0.6

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /vitvqganvae

RUN ls

COPY ./config /vitvqganvae/config
COPY ./vitvqganvae /vitvqganvae/vitvqganvae
COPY ./main.py /vitvqganvae/main.py

CMD wandb login ${WANDB_API_KEY} && accelerate launch \
    --mixed_precision=no \
    --num_processes=1 \
    --num_machines=1 \
    --dynamo_backend=no \
    main.py \
    --config config/pointcloud/mesh500/mesh500_2048_vqvae_mhvq.yaml \
    --train \
    trainer_kwargs.use_wandb_tracking=True