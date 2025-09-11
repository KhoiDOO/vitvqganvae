# Models

python -m pytest test/model/cnn/test_enc_block.py
python -m pytest test/model/cnn/test_dec_block.py

python -m pytest test/model/cnn/test_encoder.py
python -m pytest test/model/cnn/test_decoder.py

python -m pytest test/model/cnn/test_vqvae.py
python -m pytest test/model/cnn/test_flex_vqvae.py

python -m pytest test/model/vit/test_img_encoder.py
python -m pytest test/model/vit/test_img_decoder.py
python -m pytest test/model/vit/test_img_vqvae.py