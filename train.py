from config.configs_interface import configs as args
from src.trainer import WeakPMLCTrainer
# from src.logers logersimport LOGS
import os
# LOGS.init(os.path.join(args.project.PROJECT_DIR, f'{args.log.log_dir}/{args.log.log_file_name}'))

os.environ["CUDA_VISIBLE_DEVICES"] = "2"



def main():
    trainer = WeakPMLCTrainer(args)
    # # Construct category vocabulary
    trainer.category_vocabulary(top_pred_num=args.train_args.top_pred_num,
                                category_vocab_size=args.train_args.category_vocab_size)


if __name__ == "__main__":
    main()
