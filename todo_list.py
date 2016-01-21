#############
# TODO LIST #
#############
TODO_LIST="""

Sun. January 17
---------------

0. Add layer-wise KLd diagnostics to VAN tests. (DONE)
1. Add dropout to inference and generative modules and networks. (DONE)
2. Test VAN performance with increased VAE weight. (RUNNING)
3. Test VAN performance with increased lam_kld.
3. Write generative and semi-supervised MNIST tests.
4. Write SVHN semi-supervised tests.

Mon. January 18
---------------
1. Test VAN performance with increased VAE weight, and "3x3" discriminator. (DONE)
2. Test VAN performance with increased lam_kld.
3. Debug and test model save/load functionality. (DONE)
4. Debug SVHN VAE test script, to make sure it's working. (DONE)
5. Write generative and semi-supervised MNIST tests.
6. Write SVHN semi-supervised tests.

Tue. January 19
---------------
1. Test GAN performance with VAN architecture -- 5x5 and 3x3 disc. (DONE)
2. Write generative and semi-supervised MNIST tests.
3. Write SVHN semi-supervised tests.
4. Check that GAN tests produce expected results. If not, look into changing
   GenFCModule not to use resnet style, and look into reducing filter count in
   top conv module of discriminator. Also check old normalization. (DONE)

Thurs. January 21
-----------------
1. Write code for "multi-scale" reconstruction in VAE.
2. Write generative and semi-supervised MNIST tests.
3. Write SVHN semi-supervised tests.
4. Work on writing about GAN problems and VAE problems.
5. Work on an abstract, to obtain focus.

TODO: When gpus are available, train deep_dm2_dm3_match_drop01 with delayed
      onset of vae cost in the generator. This seemed to dramatically reduce
      the eventual KLd of the inference network for match_drop00_3x3_disc.





END DUMMY STRING
"""
