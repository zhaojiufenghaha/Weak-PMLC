import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from model_Bert import BERTClass, loss_fn
from process_data import data_loader
from tqdm import tqdm
import os

val_labels = []
val_outputs = []

# 模型训练函数
def train_model(start_epochs, n_epochs, valid_loss_min_input, training_loader, validation_loader, model,
                optimizer):
    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input
    best_test_acc, best_test_f1, b_pre, b_recall = 0.0, 0.0, 0.0, 0.0
    for epoch in tqdm(range(start_epochs, n_epochs + 1)):
        train_loss = 0
        valid_loss = 0

        model.train()
        print('Epoch {}: Training Start'.format(epoch))
        for batch_idx, data in enumerate(training_loader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            labels = data['label'].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids, return_dict=False)
            # print('outs',outputs)
            optimizer.zero_grad()
            loss = loss_fn(outputs, labels)
            if batch_idx % 500 == 0:
                with open('log2.txt','a',encoding='utf-8') as f:
                    f.write(f'Epoch: {epoch}, BATCH: {batch_idx}, Training Loss:  {loss.item()}'+'\n')
                print(f'Epoch: {epoch}, BATCH: {batch_idx}, Training Loss:  {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
        with open('log2.txt', 'a', encoding='utf-8') as f:
            f.write('Epoch {}: Training End'.format(epoch)+'\n')
            f.write('Epoch {}: Validation Start'.format(epoch)+'\n')
        print('Epoch {}: Training End'.format(epoch))
        print('Epoch {}: Validation Start'.format(epoch))
        # validate the model
        model.eval()

        with torch.no_grad():
            for batch_idx, data in enumerate(validation_loader, 0):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                labels = data['label'].to(device, dtype=torch.float)
                outputs = model(ids, mask, token_type_ids)

                loss = loss_fn(outputs, labels)
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
                val_labels.extend(labels.cpu().detach().numpy().tolist())
                val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
                val_predicts = (np.array(val_outputs) >= 0.5).astype(int)
                acc = accuracy_score(val_labels, val_predicts)
                with open('log2.txt', 'a', encoding='utf-8') as f:
                    f.write('Epoch:{} \t test_acc:{}\t'.format(epoch, acc)+'\n')
                print('Epoch:{} \t test_acc:{}\t'.format(epoch, acc))
                if acc > best_test_acc:
                    best_test_acc = acc
                    torch.save(model,'Best_model_Bert.pt')
                    # print('Epoch:{} \t test_acc:{}\t'.format(epoch, acc))
            with open('log2.txt', 'a', encoding='utf-8') as f:
                f.write('Epoch:{} \t best_acc:{}'.format(epoch, best_test_acc)+'\n')
                f.write('Epoch {}: Validation End'.format(epoch)+'\n')
            print('Epoch:{} \t best_acc:{}'.format(epoch, best_test_acc))
            print('Epoch {}: Validation End'.format(epoch))
            train_loss = train_loss / len(training_loader)
            valid_loss = valid_loss / len(validation_loader)
            # print training/validation statistics
            with open('log2.txt', 'a', encoding='utf-8') as f:
                f.write('Epoch: {} \t Avgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'
                  .format(epoch, train_loss, valid_loss)+'\n')
            print('Epoch: {} \t Avgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'
                  .format(epoch, train_loss, valid_loss))

            # create checkpoint variable and add important data
            # checkpoint = {
            #     'epoch': epoch + 1,
            #     'valid_loss_min': valid_loss,
            #     'state_dict': model.state_dict(),
            #     'optimizer': optimizer.state_dict()
            # }

            # save checkpoint
            # save_ckp(checkpoint, False, checkpoint_path, best_model_path)
            # save the model
            if valid_loss <= valid_loss_min:
                with open('log2.txt', 'a', encoding='utf-8') as f:
                    f.write('Validation loss decreased from {:.6f} to {:.6f}). Saving model'
                      .format(valid_loss_min, valid_loss)+'\n')
                print('Validation loss decreased from {:.6f} to {:.6f}). Saving model'
                      .format(valid_loss_min, valid_loss))
                # save checkpoint as best model
                # save_ckp(checkpoint, True, checkpoint_path, best_model_path)
                valid_loss_min = valid_loss

        print('Epoch {}  Done\n'.format(epoch))

    return model


if __name__ == '__main__':
    # 设置GPU或CPU训练
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     print(f'There are {torch.cuda.device_count()} GPU(s) available, '
    #           f'We will use the GPU: {torch.cuda.get_device_name(2)}.')
    # else:
    #     print('No GPU available, using the CPU instead.')
    # device = torch.device("cpu")
    # 加载数据集
    training_loader, validation_loader = data_loader(train_path='T_bertPolice06_train.txt', test_path='T_bertPolice06_test.txt')
    # 创建模型
    model = BERTClass()
    device = torch.device("cuda")
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-05)
    # 模型训练
    # checkpoint_path = 'current_checkpoint.pt'
    # best_model_path = 'best_model.pt'
    trained_model = train_model(1, 50, np.Inf, training_loader, validation_loader, model,
                                optimizer)
    # 模型预测指标
    val_predicts = (np.array(val_outputs) >= 0.5).astype(int)
    print('val_predicts', val_predicts)
    print('val_labels', val_labels)
    accuracy = accuracy_score(val_labels, val_predicts)
    f1_score_micro = f1_score(val_labels, val_predicts, average='micro')
    f1_score_macro = f1_score(val_labels, val_predicts, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
    with open('log2.txt', 'a', encoding='utf-8') as f:
        f.write(f"Accuracy Score = {accuracy}")
        f.write(f"F1 Score (Micro) = {f1_score_micro}")
        f.write(f"F1 Score (Macro) = {f1_score_macro}")
