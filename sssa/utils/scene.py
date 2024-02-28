
    # include ambient illumination (0.05, 0.05, 0.05)
    #sample[LIGHT] = None 
    '''
    def get_clevr_lights(
    light_jitter: float = 1.0,
    rng: np.random.RandomState = randomness.default_rng()):
    """ Create lights that match the setup from the CLEVR dataset."""
    sun = core.DirectionalLight(name="sun",
                                color=color.Color.from_name("white"), shadow_softness=0.2,
                                intensity=0.45, position=(11.6608, -6.62799, 25.8232))
    lamp_back = core.RectAreaLight(name="lamp_back",
                                    color=color.Color.from_name("white"), intensity=50.,
                                    position=(-1.1685, 2.64602, 5.81574))
    lamp_key = core.RectAreaLight(name="lamp_key",
                                    color=color.Color.from_hexint(0xffedd0), intensity=100,
                                    width=0.5, height=0.5, position=(6.44671, -2.90517, 4.2584))
    lamp_fill = core.RectAreaLight(name="lamp_fill",
                                    color=color.Color.from_hexint(0xc2d0ff), intensity=30,
                                    width=0.5, height=0.5, position=(-4.67112, -4.0136, 3.01122))
    lights = [sun, lamp_back, lamp_key, lamp_fill]

    # jitter lights
    for light in lights:
        light.position = light.position + rng.rand(3) * light_jitter
        light.look_at((0, 0, 0))
    '''
    
    # include background material (principled bsdf roughness = 1., specular=0.), 
    # color (movi_b get from metadata, movi_a #808080), friction (0.3), resitution (0.5)
    #sample[ENV] = None 
    