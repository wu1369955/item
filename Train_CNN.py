import torch.nn.functional as F
from dataset.getDataset import *
from torch import optim
from Tools import *
from networks.CNN import *
from tqdm import tqdm


def fit_epoch_train(dataset, device, batch_size, models, optimizers, lr_schedulers, statistics):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    models = [i.train() for i in models]
    C, R = models
    opt_C, opt_R = optimizers
    scheduler_C, scheduler_R = lr_schedulers
    Loss_BCE = nn.BCELoss()
    Loss_CSE = nn.CrossEntropyLoss()
    total = dataloader.__len__()
    # viz = visdom.Visdom()
    with tqdm(total=total, desc='train', postfix=dict, mininterval=0.5) as indecate:
        for img, label in dataloader:
            img = img.to(device)
            frozenModel(C)
            unfrozenModel(R)
            reconstruction = R(img)
            loss_ssim = SSIM(reconstruction, img)

            # viz.images(img, nrow=16, win='src')
            # viz.images(reconstruction, nrow=16, win='rec')

            loss_R = Loss_BCE(reconstruction, img)
            opt_R.zero_grad()
            loss_R.backward()
            opt_R.step()
            scheduler_R.step()
            frozenModel(R)
            unfrozenModel(C)
            # reconstruction doesn't need grad
            reconstruction = reconstruction.detach()
            # if pass_reconstruction is not None:
            #     reconstruction = torch.cat((reconstruction_, pass_reconstruction), dim=0)
            # else:
            #     reconstruction = reconstruction_
            # pass_reconstruction = reconstruction_[batch_size // 2:, ...]
            # pass_reconstruction.require_grad = False

            # factor_p = SSIM(img, reconstruction)
            # reconstruction = (0.8 * reconstruction + 0.2 * img)
            # # 基于AUROC的计算准则计算loss
            # logics_true = C(img)
            # True_scores = torch.ones(len(img), device=device)
            # x = torch.log_softmax(logics_true, dim=1)
            # predict_true = torch.argmax(x, dim=1)
            # mask_true = torch.zeros_like(logics_true)
            # for id, i in enumerate(predict_true):
            #     mask_true[id][i] = 1
            # mask_true = mask_true.bool()
            # scores_true = torch.masked_select(logics_true, mask_true)
            # scores_true = torch.sigmoid(scores_true)
            # loss_C_true = F.binary_cross_entropy(scores_true, True_scores)
            #
            # logics_fake = C(reconstruction)
            # mask_fake = torch.zeros_like(logics_fake)
            # Fake_scores = torch.zeros(len(reconstruction), device=device)
            # x = torch.log_softmax(logics_fake, dim=1)
            # predict_fake = torch.argmax(x, dim=1)
            # for id, i in enumerate(predict_fake):
            #     mask_fake[id][1] = 1
            # mask_fake = mask_fake.bool()
            # scores_fake = torch.masked_select(logics_fake, mask_fake)
            # scores_fake = torch.sigmoid(scores_fake)
            # loss_C_fake = Loss_BCE(scores_fake, Fake_scores)

            True_labels = torch.ones(len(img), dtype=torch.long, device=device)
            Fake_labels = torch.zeros(len(reconstruction), dtype=torch.long, device=device)

            # trueImages_labels = list(zip(list(img), list(True_labels)))
            # fakeImages_labels = list(zip(list(reconstruction), list(Fake_labels)))
            # images_labels = trueImages_labels + fakeImages_labels
            # random.shuffle(images_labels)
            # images, labels = zip(*images_labels)
            # images = [torch.unsqueeze(i, dim=0) for i in list(images)]
            # images = torch.cat(images, dim=0)
            # labels = [torch.unsqueeze(i, dim=0) for i in list(labels)]
            # labels = torch.cat(labels, dim=0)
            # logics = C(images)
            logics_true = C(img)
            # dis_true = torch.sigmoid(logics_true)
            # viz.scatter(dis_true, None, win='dis', name='True distribution',update='append')

            True_scores = torch.ones(len(img), device=device)
            x = torch.softmax(logics_true, dim=1)
            predict_true = torch.argmax(x, dim=1)
            mask_true = torch.zeros_like(logics_true)
            for id, i in enumerate(predict_true):
                mask_true[id][i] = 1
            mask_true = mask_true.bool()
            scores_true = torch.masked_select(logics_true, mask_true)
            scores_true = torch.sigmoid(scores_true)
            loss_C_true = F.binary_cross_entropy(scores_true, True_scores)

            logics_fake = C(reconstruction)
            # dis_fake = torch.sigmoid(logics_fake)
            # viz.scatter(dis_fake, None, win='dis', name='Fake distribution',update='append')

            mask_fake = torch.zeros_like(logics_fake)
            Fake_scores = torch.zeros(len(reconstruction), device=device)
            x = torch.softmax(logics_fake, dim=1)
            predict_fake = torch.argmax(x, dim=1)
            for id, i in enumerate(predict_fake):
                mask_fake[id][1] = 1
            mask_fake = mask_fake.bool()
            scores_fake = torch.masked_select(logics_fake, mask_fake)
            scores_fake = torch.sigmoid(scores_fake)
            loss_C_fake = Loss_BCE(scores_fake, Fake_scores)

            trueLogics_labels = list(zip(list(logics_true), list(True_labels)))
            fakeLogics_labels = list(zip(list(logics_fake), list(Fake_labels)))
            logics_labels = trueLogics_labels + fakeLogics_labels
            random.shuffle(logics_labels)
            logics, labels = zip(*logics_labels)
            logics = [torch.unsqueeze(i, dim=0) for i in list(logics)]
            logics = torch.cat(logics, dim=0)
            labels = [torch.unsqueeze(i, dim=0) for i in list(labels)]
            labels = torch.cat(labels, dim=0)
            # x_labels = labels + 1
            # viz.scatter(torch.softmax(logics, dim=1), Y=x_labels, win='Train',
            #             opts=dict(title='Train', legend=['Fake data', 'True data']))

            loss_C = Loss_CSE(logics, labels)
            loss_C = loss_C + loss_C_true + loss_C_fake
            opt_C.zero_grad()
            loss_C.backward()
            opt_C.step()
            scheduler_C.step()

            statistics[-1] += loss_ssim
            statistics[0] += 1
            indecate.update(1)

    statistics[-1] /= statistics[0]


def fit_epoch_val_test(dataset, device, batch_size, models, normalClass, statistics):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    models = [i.eval() for i in models]
    C, R = models
    frozenModel(C)
    # viz = visdom.Visdom()
    testStep = total_recall = total_precision = total_accuracy = total_f1 = total_AUROC = total_AUPR = 0
    for img, label in dataloader:
        img = img.to(device)
        labels = [0 if i != normalClass else 1 for i in label]

        y = set(label.numpy())
        legend = ['Fake data ' + str(i) if i != normalClass else 'True data ' + str(i) for i in y]
        y = {i: id + 1 for id, i in enumerate(y)}
        x_label = [y[i] for i in label.numpy()]

        logics = C(img)
        # viz.scatter(torch.softmax(logics, dim=1), Y=x_label, win='Test', opts=dict(title='Test', legend=legend))
        x = torch.softmax(logics, dim=0)
        predict = torch.argmax(x, dim=1)
        mask = torch.zeros_like(logics)
        for id, i in enumerate(predict):
            mask[id][1] = 1
        mask = mask.bool()
        scores = torch.masked_select(logics, mask)
        scores = torch.sigmoid(scores)
        predict = predict.cpu().detach().numpy()
        scores = scores.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(labels, predict)
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(labels, predict, zero_division=0)
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
        AUROC = metrics.auc(fpr, tpr)

        testStep += 1
        total_recall += np.max(recall)
        total_precision += np.max(precision)
        total_accuracy += accuracy
        total_f1 += np.max(f1)
        total_AUROC += AUROC if AUROC is not float('nan') else 0

    recall = total_recall / testStep
    precision = total_precision / testStep
    accuracy = total_accuracy / testStep
    f1 = total_f1 / testStep
    AUROC = total_AUROC / testStep
    AUPR = total_AUPR / testStep

    return f1, recall, precision, accuracy, AUROC, AUPR


def train_scheduler(epochs, scheduler, preparations):
    classes, test_per_epoch, proportions, times = scheduler
    for inlier in classes:
        datasetName, root, transform, device, batch_size, models, optimizers, lr_schedulers, log = preparations
        C, R = models
        C.weights_init()
        R.weights_init()
        scheduler_C, scheduler_R = lr_schedulers
        trainStep = best_f1 = best_auroc = best_f1_epoch = best_auroc_epoch = best_acc = best_acc_epoch = ssim = 0
        statistics = [trainStep, best_f1, best_auroc, best_f1_epoch, best_auroc_epoch, best_acc, best_acc_epoch, ssim]
        trainDataset = getDataset(datasetName=datasetName, root=root, train=True, download=True, normalClass=inlier,
                                  transform=transform)
        for epoch in range(epochs):
            # print(f'train_epoch:{epoch + 1} normal class:{inlier}')
            fit_epoch_train(trainDataset, device, batch_size, models, optimizers, lr_schedulers, statistics)
            if epoch % test_per_epoch == 0:
                # print(f'val_epoch:{epoch + 1} normal class:{inlier},proportion:5')
                testDataset = getDataset(datasetName=datasetName, root=root, train=False, download=True,
                                         normalClass=inlier, transform=transform, proportion=5)
                f1, recall, precision, accuracy, auroc, aupr = fit_epoch_val_test(testDataset, device, batch_size,
                                                                                  models, inlier, statistics)
                # print(
                #     f'\nepoch:{epoch} f1:{f1:.4f} auroc:{auroc:.4f} accuracy:{accuracy:.4f} recall:{recall:.4f} precision:{precision:.4f}\n')
                log.write(
                    f'normal class:{inlier} epoch:{epoch} f1:{f1:.4f} auroc:{auroc:.4f} accuracy:{accuracy:.4f} recall:{recall:.4f} precision:{precision:.4f}\n')
                log.flush()
                if auroc >= best_auroc:
                    best_auroc_epoch = epoch
                    best_auroc = auroc
                    torch.save(C.state_dict(), f'best_auroc_weights.pth')
                if f1 >= best_f1:
                    best_f1_epoch = epoch
                    best_f1 = f1
                    torch.save(C.state_dict(), f'best_f1_weights.pth')
                if accuracy >= best_acc:
                    best_acc_epoch = epoch
                    best_acc = accuracy
                    torch.save(C.state_dict(), f'best_acc_weights.pth')
                # print(f'\nbest_auroc_epoch:{best_auroc_epoch} best_auroc:{best_auroc:.4f} '
                #       f'best_f1_epoch:{best_f1_epoch} best_f1:{best_f1:.4f} '
                #       f'best_accuracy_epoch:{best_acc_epoch} best_accuracy:{best_acc:.4f}\n')

        for proportion in proportions:

            ave_f1 = ave_recall = ave_precision = ave_accuracy = ave_auroc = ave_aupr = 0
            C, R = models
            C.load_state_dict(torch.load('best_auroc_weights.pth'))
            models = [C, R]
            for i in range(times):
                testDataset = getDataset(datasetName=datasetName, root=root, train=False, download=True,
                                         normalClass=inlier, transform=transform, proportion=proportion)
                t_f1, t_recall, t_precision, t_accuracy, t_auroc, t_aupr = fit_epoch_val_test(testDataset, device,
                                                                                              batch_size, models,
                                                                                              inlier, statistics)
                ave_f1 += t_f1
                ave_recall += t_recall
                ave_precision += t_precision
                ave_accuracy += t_accuracy
                ave_auroc += t_auroc
                ave_aupr += t_aupr

            ave_f1 /= times
            ave_recall /= times
            ave_precision /= times
            ave_accuracy /= times
            ave_auroc /= times
            ave_aupr /= times
            print(f'inlier class : {inlier},test proportion : {proportion} '
                  f'f1:{ave_f1:.4f} Recall:{ave_recall:.4f} Precision:{ave_precision:.4f} Accuracy:{ave_accuracy:.4f} '
                  f'AUROC:{ave_auroc:.4f} AUPR:{ave_aupr:.4f}')
            log.write(f'Result inlier class : {inlier},test proportion : {proportion} '
                      f'f1:{ave_f1:.4f} Recall:{ave_recall:.4f} Precision:{ave_precision:.4f} Accuracy:{ave_accuracy:.4f} '
                      f'AUROC:{ave_auroc:.4f} AUPR:{ave_aupr:.4f}\n')
            log.flush()


def train():
    datasetName = 'MNIST'
    root = './dataset'
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        lambda x: x.convert('RGB'),
        transforms.ToTensor(),
    ])
    device = torch.device('cpu')
    batch_size = 128
    lr = 1e-3
    epochs = 30

    C = ResNet18(2)
    R = Reconstructor(32)
    C.to(device)
    R.to(device)
    models = [C, R]

    opt_C = optim.Adam(C.parameters(), lr=lr)
    opt_R = optim.Adam(R.parameters(), lr=lr)
    optimizers = [opt_C, opt_R]

    gama = 0.9
    scheduler_C = optim.lr_scheduler.StepLR(opt_C, step_size=10, gamma=gama)
    scheduler_R = optim.lr_scheduler.StepLR(opt_R, step_size=30, gamma=gama)
    lr_schedulers = [scheduler_C, scheduler_R]

    classes = [4]
    test_per_epoch = 1
    proportions = [i for i in range(1, 10)]
    times = 10

    scheduler = [classes, test_per_epoch, proportions, times]
    log = open('myMNIST_4_log.txt', mode='w+')
    preparations = [datasetName, root, transform, device, batch_size, models, optimizers, lr_schedulers, log]
    train_scheduler(epochs, scheduler, preparations)
    log.close()


if __name__ == '__main__':
    train()
