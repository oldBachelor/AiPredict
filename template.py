
def set_template(args):
    if args.template.find('Mymodel') >= 0:
        args.dataset = 'elect'
        args.model = 'Mymodel4'
        args.patch_size = 41
        args.lr = 1e-1