from data import *
from net import *
from lib import *
from easydl import *
import datetime
from tqdm import tqdm
if is_in_notebook():
    from tqdm import tqdm_notebook as tqdm
from torch import optim
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import os
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
import seaborn as sns
cudnn.benchmark = True
cudnn.deterministic = True

seed_everything()

if args.misc.gpus < 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    gpu_ids = []
    output_device = torch.device('cpu')
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.misc.gpu_id
    gpu_ids = args.misc.gpu_id_list

if args.test.test_only:
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    gpu_ids = [0]

now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')

log_dir = f'{args.log.root_dir}/{now}'

logger = SummaryWriter(log_dir)

with open(join(log_dir, 'config.yaml'), 'w') as f:
    f.write(yaml.dump(save_config))

model_dict = {
    'resnet50': ResNet50Fc,
    'vgg16': VGG16Fc
}
batch_size = args.data.dataloader.batch_size

def sns_plot(para_source, para_target, source_shared_index, source_private_index, target_shared_index, target_private_index, global_step, name, log=False):
    source_share = torch.index_select(para_source, dim=0, index=source_shared_index).flatten().cpu().detach().numpy()
    source_private = torch.index_select(para_source, dim=0, index=source_private_index).flatten().cpu().detach().numpy()
    target_share = torch.index_select(para_target, dim=0, index=target_shared_index).flatten().cpu().detach().numpy()
    target_private = torch.index_select(para_target, dim=0, index=target_private_index).flatten().cpu().detach().numpy()
    if log:
        logger.add_scalar('weight/source_share_weight', source_share.mean(), global_step)
        logger.add_scalar('weight/source_private_weight', source_private.mean(), global_step)
        logger.add_scalar('weight/target_share_weight', target_share.mean(), global_step)
        logger.add_scalar('weight/target_private_weight', target_private.mean(), global_step)
    sns.set()
    sns.kdeplot(source_share, cut=0, label='source share')
    sns.kdeplot(source_private, cut=0, label='source private')
    sns.kdeplot(target_share, cut=0, label='target share')
    sns.kdeplot(target_private, cut=0, label='target private')
    plt.legend()
    plt.savefig(join(log_dir, name ))
    plt.close()


def label_to_RGB(label):
    color = np.zeros((len(label), 3))
    for index in range(len(label)):
        if label[index] == 0:
            color[index] = np.array([1, 0, 0])   # red: target samples wrong aligned to known classes
        if label[index] == 1:
            color[index] = np.array([0, 1, 0])   # green: source and target samples with shared labels
        if label[index] == 2:
            color[index] = np.array([0, 0, 1])   # blue: source samples with source private labels
        if label[index] == 3:
            color[index] = np.array([0, 0, 0])   # black: target samples with target private labels (refer as the unknown category)
    return color

class TotalNet(nn.Module):
    def __init__(self):
        super(TotalNet, self).__init__()
        self.feature_extractor = model_dict[args.model.base_model](args.model.pretrained_model)
        self.generator = GeneratedNetwork(256)
        classifier_output_dim = len(source_classes)
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim)
        self.domain_discriminator = AdversarialNetwork(256)

    def forward(self, x):
        f = self.feature_extractor(x)
        f_generated = self.generator(x)
        f, _, __, y, y_ = self.classifier(f)
        d = self.domain_discriminator(_)
        return y, y_ , d, d_


totalNet = TotalNet()
feature_extractor = nn.DataParallel(totalNet.feature_extractor.cuda(), device_ids=gpu_ids).train(True)
classifier = nn.DataParallel(totalNet.classifier.cuda(), device_ids=gpu_ids).train(True)
domain_discriminator = nn.DataParallel(totalNet.domain_discriminator.cuda(), device_ids=gpu_ids).train(True)

if args.test.test_only:
    assert os.path.exists(args.test.resume_file)
    data = torch.load(open(args.test.resume_file, 'rb'))
    feature_extractor.load_state_dict(data['feature_extractor'])
    classifier.load_state_dict(data['classifier'])
    domain_discriminator.load_state_dict(data['domain_discriminator'])

    counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]
    with TrainingModeManager([feature_extractor, classifier], train=False) as mgr, \
            Accumulator(['feature', 'predict_prob', 'label']) as target_accumulator, \
            torch.no_grad():
        label_list = np.zeros(len(target_test_ds))
        mean_max_predict_prob = np.zeros(1)
        for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing ')):
            im = im.cuda()
            label = label.cuda()

            feature = feature_extractor.forward(im)
            feature, fc1, before_softmax, predict_prob = classifier.forward(feature)
            mean_max_predict_prob = (mean_max_predict_prob * i + variable_to_numpy(torch.mean(predict_prob.max(1)[0])))/(i + 1)
            label_list[i * batch_size:i * batch_size + len(label)] = label

            for name in target_accumulator.names:
                globals()[name] = variable_to_numpy(globals()[name])

            target_accumulator.updateData(globals())

    for x in target_accumulator:
        globals()[x] = target_accumulator[x]

    counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]

    for (each_predict_prob, each_label) in zip(predict_prob, label):
        each_pred_id = np.argmax(each_predict_prob)
        if each_label in source_classes:
            counters[each_label].Ntotal += 1.0
            if each_pred_id == each_label and np.max(each_predict_prob) >= 0.5: #
                counters[each_label].Ncorrect += 1.0
        else:
            counters[-1].Ntotal += 1.0
            if np.max(each_predict_prob) < 0.5:
                counters[-1].Ncorrect += 1.0

    acc_tests = [x.reportAccuracy() for x in counters if not np.isnan(x.reportAccuracy())]
    acc_test = torch.ones(1, 1) * np.mean(acc_tests)
    acc_known = [x.reportAccuracy() for x in counters[:-1] if not np.isnan(x.reportAccuracy())]
    acc_known = torch.ones(1, 1) * np.mean(acc_known)
    acc_unknown = counters[-1].Ncorrect / counters[-1].Ntotal
    acc_unknown = torch.ones(1, 1) * acc_unknown
    print(f'test accuracy is {acc_test.item()}\n', f'the known accuracy is {acc_known.item()}\n', f'the unknown accuracy is {acc_unknown.item()}\n')

    known_num = np.where(label_list<args.data.dataset.n_share, 1, 0).sum()
    feature_list = feature[:known_num]
    Y = TSNE(n_jobs=4).fit_transform(feature_list)
    plt.scatter(Y[:, 0], Y[:, 1], s=10, c=label_list[:known_num], marker='^')
    plt.savefig('./log/{}/test_known_distribution.png'.format(now))
    plt.close()
    exit(0)

# ===================optimizer
scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
if args.misc.gpus > 1:
    optimizer_finetune = OptimWithSheduler(
        optim.SGD(feature_extractor.module.parameters(), lr=args.train.lr / 10.0, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
        scheduler)
    optimizer_cls = OptimWithSheduler(
        optim.SGD(classifier.module.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
        scheduler)
    optimizer_domain_discriminator = OptimWithSheduler(
        optim.SGD(domain_discriminator.module.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
        scheduler)
else:
    optimizer_finetune = OptimWithSheduler(
        optim.SGD(feature_extractor.parameters(), lr=args.train.lr / 10.0, weight_decay=args.train.weight_decay,momentum=args.train.momentum, nesterov=True),scheduler)
    optimizer_cls = OptimWithSheduler(
        optim.SGD(classifier.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay,momentum=args.train.momentum, nesterov=True),scheduler)
    optimizer_domain_discriminator = OptimWithSheduler(
        optim.SGD(domain_discriminator.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay,momentum=args.train.momentum, nesterov=True),scheduler)

global_step = 0
records = 0
best_acc = 0

total_steps = tqdm(range(args.train.min_step),desc='global step')
epoch_id = 0
class_temperture = torch.zeros(len(source_classes),1).cuda()

while global_step < args.train.min_step:

    iters = tqdm(zip(source_train_dl, target_train_dl), desc=f'epoch {epoch_id} ', total=min(len(source_train_dl), len(target_train_dl)))
    epoch_id += 1

    for i, ((im_source, label_source), (im_target, label_target)) in enumerate(iters):

        save_label_target = label_target  # for debug usage

        label_source = Variable(label_source.cuda())
        label_target = Variable(label_target.cuda())
        label_target = torch.zeros_like(label_target)

        # =========================forward pass
        im_source = Variable(im_source.cuda())
        im_target = Variable(im_target.cuda())

        fc1_s = feature_extractor.forward(im_source)
        fc1_t = feature_extractor.forward(im_target)

        fc1_s, feature_source, fc2_s, predict_prob_source = classifier.forward(fc1_s)
        fc1_t, feature_target, fc2_t, predict_prob_target = classifier.forward(fc1_t)

        domain_prob_source = domain_discriminator.forward(feature_source)
        domain_prob_target = domain_discriminator.forward(feature_target)

        counter = AccuracyCounter()
        counter.addOneBatch(variable_to_numpy(one_hot(label_source, len(source_classes))),variable_to_numpy(predict_prob_source))
        acc_train = torch.tensor([counter.reportAccuracy()]).cuda()

        with torch.no_grad():
            target_share_weight = torch.zeros(batch_size,1).cuda()
            source_share_weight = torch.zeros(batch_size,1).cuda()
            pred_weighted_source = torch.zeros(batch_size, 1).cuda()
            target_margin = torch.zeros(len(source_classes), 1).cuda()
            target_pseudo_label = predict_prob_target.max(1)[1]
            num_per_class = torch.zeros(len(source_classes), 1).cuda()
            sorted_pred_target = torch.sort(predict_prob_target, dim=1, descending=True)[0]
            for index in range(batch_size):
                target_margin[target_pseudo_label[index],0] += sorted_pred_target[index,0] - sorted_pred_target[index,1]
                num_per_class[target_pseudo_label[index],0] +=1
            target_pred_per_label = ((class_temperture * records) + torch.div(target_margin, num_per_class + 1e-6)) / (records + 1)
            if acc_train > 0.6:
                class_temperture = target_pred_per_label.detach()
                records += 1
            for index in range(batch_size):
                source_share_weight[index, 0] = target_pred_per_label[label_source[index]] #- domain_prob_source[index]
                target_share_weight[index, 0] = predict_prob_target.max(1)[0][index]
                pred_weighted_source[index, 0] = target_pred_per_label[label_source[index]]

            source_shared_label = torch.lt(label_source, args.data.dataset.n_share).view(batch_size,1).float()
            source_shared_index = torch.nonzero(source_shared_label.flatten()).flatten()
            source_private_label = torch.ge(label_source,args.data.dataset.n_share).view(batch_size,1).float()
            source_private_index = torch.nonzero(source_private_label.flatten()).flatten()
            target_shared_label = torch.lt(label_target,args.data.dataset.n_share).view(batch_size,1).float()
            target_shared_index = torch.nonzero(target_shared_label.flatten()).flatten()
            target_private_label = torch.ge(label_target,args.data.dataset.n_share + args.data.dataset.n_source_private).view(batch_size,1).float()
            target_private_index = torch.nonzero(target_private_label.flatten()).flatten()

        source_share_weight_normalized = normalize_weight(source_share_weight, cut=args.train.cut)
        target_share_weight_normalized = normalize_weight(target_share_weight, cut=0)

        # ==============================compute loss
        cls_s = nn.CrossEntropyLoss()(fc2_s, label_source)

        domain_adv_loss = torch.zeros(1, 1).cuda()
        tmp = source_share_weight_normalized * nn.BCELoss(reduction='none')(domain_prob_source,torch.ones_like(domain_prob_source))
        domain_adv_loss += torch.mean(tmp, dim=0, keepdim=True)
        tmp = target_share_weight_normalized * nn.BCELoss(reduction='none')(domain_prob_target,torch.zeros_like(domain_prob_target))
        domain_adv_loss += torch.mean(tmp, dim=0, keepdim=True)

        if acc_train > 0.6:
            with OptimizerManager(
                    [optimizer_finetune, optimizer_cls, optimizer_domain_discriminator]):
                loss = cls_s + domain_adv_loss
                loss.backward()
        else:
            with OptimizerManager(
                    [optimizer_finetune, optimizer_cls]):
                loss = cls_s
                loss.backward()


        global_step += 1
        total_steps.update()

        if global_step % args.log.log_interval == 0:
            logger.add_scalar('loss/domain_adv_loss', domain_adv_loss, global_step)
            logger.add_scalar('loss/cls_s', cls_s, global_step)
            logger.add_scalar('acc_train', acc_train, global_step)

        if global_step % args.test.test_interval == 0:
            counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]
            with TrainingModeManager([feature_extractor, classifier], train=False) as mgr, \
                 Accumulator(['feature_test', 'predict_prob_test', 'label']) as target_accumulator, \
                 torch.no_grad():
                mean_max_predict_prob = np.zeros(1)
                for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing ')):
                    im = im.cuda()
                    label = label.cuda()

                    feature_test = feature_extractor.forward(im)
                    feature_test, feature_short, before_softmax, predict_prob_test = classifier.forward(feature_test)
                    mean_max_predict_prob = (mean_max_predict_prob * i + variable_to_numpy(torch.mean(predict_prob_test.max(1)[0]))) / (i + 1)

                    for name in target_accumulator.names:
                        globals()[name] = variable_to_numpy(globals()[name])

                    target_accumulator.updateData(globals())

            for x in target_accumulator:
                globals()[x] = target_accumulator[x]


            counters = [AccuracyCounter() for x in range(len(source_classes) + 1)]

            for (each_predict_prob, each_label) in zip(predict_prob_test, label):
                each_pred_id = np.argmax(each_predict_prob)
                if each_label in source_classes:
                    counters[each_label].Ntotal += 1.0
                    if each_pred_id == each_label and np.max(each_predict_prob) >= 0.5:
                        counters[each_label].Ncorrect += 1.0
                else:
                    counters[-1].Ntotal += 1.0
                    if np.max(each_predict_prob) < 0.5:
                        counters[-1].Ncorrect += 1.0

            acc_tests = [x.reportAccuracy() for x in counters if not np.isnan(x.reportAccuracy())]
            acc_test = torch.ones(1, 1) * np.mean(acc_tests)
            acc_known = [x.reportAccuracy() for x in counters[:-1] if not np.isnan(x.reportAccuracy())]
            acc_known = torch.ones(1, 1) * np.mean(acc_known)
            acc_unknown = counters[-1].Ncorrect / counters[-1].Ntotal
            acc_unknown = torch.ones(1, 1) * acc_unknown

            logger.add_scalar('acc/acc_test', acc_test, global_step)
            logger.add_scalar('acc/acc_known', acc_known, global_step)
            logger.add_scalar('acc/acc_unknown', acc_unknown, global_step)
            logger.add_scalar('weight/threshold', np.where(mean_max_predict_prob<0.5,mean_max_predict_prob,0.5), global_step)
            clear_output()

            data = {
                "feature_extractor": feature_extractor.state_dict(),
                'classifier': classifier.state_dict(),
                'domain_discriminator': domain_discriminator.state_dict() if not isinstance(domain_discriminator, Nonsense) else 1.0,
            }

            if acc_test > best_acc:
                best_acc = acc_test
                with open(join(log_dir, 'best.pkl'), 'wb') as f:
                    torch.save(data, f)

            with open(join(log_dir, 'current.pkl'), 'wb') as f:
                torch.save(data, f)

            # sns_plot(source_share_weight_normalized, target_share_weight_normalized, source_shared_index, source_private_index,
            #          target_shared_index, target_private_index, global_step, name='step{}_w.png'.format(global_step), log=True)
