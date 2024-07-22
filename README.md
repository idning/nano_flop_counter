# nano_flop_counter

A very simple flops counter output fqn style stats

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/idning/121ec7e37634091353c946bb01859da1/nanoflopcounter.ipynb)


# Usage - ResNet example

    from torchvision import models as torchvision_models
    resnet18 = torchvision_models.resnet18()
    a = torch.randn((1, 3, 224, 224), requires_grad=True)
    with NanoFlopCounter(resnet18) as nfc:
        resnet18(a)
    print(tabulate(nfc.df(), headers='keys', tablefmt='plain', intfmt=","))

       module                           flops      params
     0  ~                        3,628,146,688  11,689,512
     1  ~.conv1                    236,027,904       9,408
     2  ~.layer1                   924,844,032     147,968
     3  ~.layer1.0                 462,422,016      73,984
     4  ~.layer1.0.conv1           231,211,008      36,864
     5  ~.layer1.0.conv2           231,211,008      36,864
     6  ~.layer1.1                 462,422,016      73,984
     7  ~.layer1.1.conv1           231,211,008      36,864
     8  ~.layer1.1.conv2           231,211,008      36,864
     9  ~.layer2                   822,083,584     525,568
    10  ~.layer2.0                 359,661,568     230,144
    11  ~.layer2.0.conv1           115,605,504      73,728
    12  ~.layer2.0.conv2           231,211,008     147,456
    13  ~.layer2.0.downsample       12,845,056       8,448
    14  ~.layer2.0.downsample.0     12,845,056       8,192
    15  ~.layer2.1                 462,422,016     295,424
    16  ~.layer2.1.conv1           231,211,008     147,456
    17  ~.layer2.1.conv2           231,211,008     147,456
    18  ~.layer3                   822,083,584   2,099,712
    19  ~.layer3.0                 359,661,568     919,040
    20  ~.layer3.0.conv1           115,605,504     294,912
    21  ~.layer3.0.conv2           231,211,008     589,824
    22  ~.layer3.0.downsample       12,845,056      33,280
    23  ~.layer3.0.downsample.0     12,845,056      32,768
    24  ~.layer3.1                 462,422,016   1,180,672
    25  ~.layer3.1.conv1           231,211,008     589,824
    26  ~.layer3.1.conv2           231,211,008     589,824
    27  ~.layer4                   822,083,584   8,393,728
    28  ~.layer4.0                 359,661,568   3,673,088
    29  ~.layer4.0.conv1           115,605,504   1,179,648
    30  ~.layer4.0.conv2           231,211,008   2,359,296
    31  ~.layer4.0.downsample       12,845,056     132,096
    32  ~.layer4.0.downsample.0     12,845,056     131,072
    33  ~.layer4.1                 462,422,016   4,720,640
    34  ~.layer4.1.conv1           231,211,008   2,359,296
    35  ~.layer4.1.conv2           231,211,008   2,359,296
    36  ~.fc                         1,024,000     513,000


# Original torch.utils.flop_counter.FlopCounterMode

    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(display=False) as fc:
        resnet18(a)
    print(fc.get_table())

    Module                    FLOP    % Total
    -------------------  ---------  ---------
    Global               3628.147M    100.00%
     - aten.convolution  3627.123M     99.97%
     - aten.addmm           1.024M      0.03%

# Usage - NanoGPT Example

    !git clone https://github.com/karpathy/nanoGPT
    import sys
    sys.path.append("nanoGPT")


    import torch

    from model import GPT, GPTConfig
    gpt2=dict(n_layer=12, n_head=12, n_embd=768)  # 124M params
    config = GPTConfig(**gpt2)
    model = GPT(config)

    x = torch.randint(high=100, size=(2, 1024))
    with NanoFlopCounter(model) as mode:
        model(x)
    print(tabulate(mode.report(), headers='keys', tablefmt='plain', intfmt=","))

    number of parameters: 123.69M
        module                                    flops       params
     0  ~                               348,086,206,464  124,475,904
     1  ~.transformer                   347,930,099,712  124,475,904
     2  ~.transformer.h                 347,930,099,712   85,054,464
     3  ~.transformer.h.0                28,994,174,976    7,087,872
     4  ~.transformer.h.0.attn            9,663,676,416    2,362,368
     5  ~.transformer.h.0.attn.c_attn     7,247,757,312    1,771,776
     6  ~.transformer.h.0.attn.c_proj     2,415,919,104      590,592
     7  ~.transformer.h.0.mlp            19,327,352,832    4,722,432
     8  ~.transformer.h.0.mlp.c_fc        9,663,676,416    2,362,368
     9  ~.transformer.h.0.mlp.c_proj      9,663,676,416    2,360,064
    10  ~.transformer.h.1                28,994,174,976    7,087,872
    11  ~.transformer.h.1.attn            9,663,676,416    2,362,368
    12  ~.transformer.h.1.attn.c_attn     7,247,757,312    1,771,776
    13  ~.transformer.h.1.attn.c_proj     2,415,919,104      590,592
    14  ~.transformer.h.1.mlp            19,327,352,832    4,722,432
    15  ~.transformer.h.1.mlp.c_fc        9,663,676,416    2,362,368
    16  ~.transformer.h.1.mlp.c_proj      9,663,676,416    2,360,064
    ...

# Original torch.utils.flop_counter.FlopCounterMode

    with FlopCounterMode(display=False) as fc:
        model(x)
    print(fc.get_table())

    Module             FLOP    % Total
    -------------  --------  ---------
    Global         348.047B    100.00%
     - aten.addmm  347.892B     99.96%
     - aten.mm       0.155B      0.04%

