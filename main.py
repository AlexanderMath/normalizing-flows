import torch
import argparse
import sys


def sample(net, batch_size, device):
    z = torch.randn((batch_size, 3, 32, 32), dtype=torch.float32, device=device)

    for i in range(batch_size):  # sample with different variances. 
      scale = (i+1) / (batch_size +1)
      z[i] *= scale

    x, _ = net(z, reverse=True)
    x = torch.sigmoid(x)
    return x

global_step = 0
@torch.enable_grad()
def train_func(epoch, net, trainloader, device, optimizer, loss_fn, max_grad_norm, args, scheduler=None):
  global global_step
  print('\nEpoch: %d' % epoch)
  if epoch == 0: 
    print(vars(args))

  #print("nloss", args.nloss, "\twloss", args.wloss)
  net.train()

  for i, (x, _) in enumerate(trainloader):
    loss = 0
    x = x.to(device)
    optimizer.zero_grad()
 
    z, sldj = net(x, reverse=False)

    loss = loss_fn(z, sldj)

    loss.backward()
    optimizer.step()

    print("\r[%i / %i] loss: \t%.4f"%(i, len(trainloader), loss.item()), end="", flush=True)

    if not scheduler is None: 
      scheduler.step(global_step)
      global_step += x.size(0)
      print("lr=%.8f\t"%( optimizer.param_groups[0]['lr']), end="", flush=True)


   # Save samples and data
    '''num_samples = 25
    images = sample(net, num_samples, device)
    os.makedirs('samples/%s/'%title, exist_ok=True)
    images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    torchvision.utils.save_image(images_concat, 'samples/%s/%i_%i.png'%(title, epoch, i+1))'''



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train RealNVP, Flow or Glow on CIFAR-10')

  parser.add_argument('--model',      default="realnvp", type=str, help='Which model to train, currently supports realnvp, flowpp and glow. ')
  parser.add_argument('--batch_size', default=64, type=int, help='Batch size')

  args = parser.parse_args()

  if args.model == "realnvp": 
    sys.path.append("realnvp")
    from train import default_args
    default_args(train_func) 

  elif args.model == "glow": 
    sys.path.append("glow")
    from train import default_args
    default_args(train_func) 

  elif args.model == "flowpp": 
    sys.path.append("flowpp")
    from train import default_args
    default_args(train_func) 

  elif args.model == "iresnet": 
    sys.path.append("iresnet")
    from CIFAR_main import main
    main()




    


