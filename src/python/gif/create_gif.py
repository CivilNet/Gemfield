import imageio
 
def create_gif(image_list, gif_name):
 
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    # Save them as frames into a gif 
    imageio.mimsave(gif_name, frames, 'GIF', duration = 0.1)
 
    return
import glob 
def main():
    image_list = glob.glob("*.jpg") 
    image_list = sorted(image_list)
    gif_name = 'created_gif.gif'
    print([x for x in image_list if int(x.split('.')[0]) % 4 == 0])
    create_gif([x for x in image_list if int(x.split('.')[0]) % 4 == 0][:16], gif_name)
 
if __name__ == "__main__":
    main()
