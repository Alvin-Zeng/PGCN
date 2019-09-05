import os
import shutil
import time


class Recorder(object):
    def __init__(self, snapshot_pref, exclude_dirs=None, max_file_size=10):
        """
        :param snapshot_pref: The dir you want to save the backups
        :param exclude_dirs: The dir name you want to exclude; eg ["results", "data"]
        :param max_file_size: The minimum size of backups file; unit is MB
        """
        date = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        if not os.path.isdir(snapshot_pref):
            os.mkdir(snapshot_pref)
        self.save_path = snapshot_pref
        self.log_file = self.save_path + "/log.txt"
        self.readme = self.save_path + "/README.md"
        self.opt_file = self.save_path + "/opt.log"
        self.code_path = os.path.join(self.save_path, "code_{}/".format(date))
        self.exclude_dirs = exclude_dirs
        self.max_file_size = max_file_size
        # self.weight_folder = self.save_path + "weight/"
        # self.weight_fig_folder = self.save_path + "weight_fig/"
        # if os.path.isfile(self.log_file):
        #     os.remove(self.log_file)
        if os.path.isfile(self.readme):
            os.remove(self.readme)
        if not os.path.isdir(self.code_path):
            os.mkdir(self.code_path)
        self.copy_code(dst=self.code_path)
        """if os.path.isdir(self.weight_folder):
            shutil.rmtree(self.weight_folder, ignore_errors=True)
        os.mkdir(self.weight_folder)
        if os.path.isdir(self.weight_fig_folder):
            shutil.rmtree(self.weight_fig_folder, ignore_errors=True)
        os.mkdir(self.weight_fig_folder)"""

        print ("|===>Backups will be saved at", self.save_path)

    def copy_code(self, src="./", dst="./code/"):
        # import time
        # date = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        # if os.path.isdir(dst):
        #     dst = "/".join(dst.split("/")[:-1]) + "_" + date + "/"
        start_time = time.time()
        file_abs_list = []
        src_abs = os.path.abspath(src)
        for root, dirs, files in os.walk(src_abs):
            exclude_flag = True in [root.find(exclude_dir)>=0 for exclude_dir in self.exclude_dirs]
            if not exclude_flag:
                for name in files:
                    file_abs_list.append(root + "/" + name)

        for file_abs in file_abs_list:
            file_split = file_abs.split("/")[-1].split('.')
            # if len(file_split) >= 2 and file_split[1] == "py":
            if os.path.getsize(file_abs) / 1024 / 1024 < self.max_file_size and not file_split[-1] == "pyc":
                src_file = file_abs
                dst_file = dst + file_abs.replace(src_abs, "")
                if not os.path.exists(os.path.dirname(dst_file)):
                    os.makedirs(os.path.dirname(dst_file))
                shutil.copyfile(src=src_file, dst=dst_file)
                try:
                    shutil.copyfile(src=src_file, dst=dst_file)
                except:
                    print("copy file error")
        print("|===>Backups using time: %.3f s"%(time.time() - start_time))

    def writeopt(self, opt):
        with open(self.opt_file, "w") as f:
            for k, v in opt.__dict__.items():
                f.write(str(k)+": "+str(v)+"\n")

    def writelog(self, input_data):
        txt_file = open(self.log_file, 'a+')
        txt_file.write(str(input_data) + "\n")
        txt_file.close()

    def writereadme(self, input_data):
        txt_file = open(self.readme, 'a+')
        txt_file.write(str(input_data) + "\n")
        txt_file.close()

    def gennetwork(self, var):
        self.graph.draw(var=var)

    def savenetwork(self):
        self.graph.save(file_name=self.save_path+"network.svg")

    """def writeweights(self, input_data, block_id, layer_id, epoch_id):
        txt_path = self.weight_folder + "conv_weight_" + str(epoch_id) + ".log"
        txt_file = open(txt_path, 'a+')
        write_str = "%d\t%d\t%d\t" % (epoch_id, block_id, layer_id)
        for x in input_data:
            write_str += str(x) + "\t"
        txt_file.write(write_str+"\n")

    def drawhist(self):
        drawer = DrawHistogram(txt_folder=self.weight_folder, fig_folder=self.weight_fig_folder)
        drawer.draw()"""
