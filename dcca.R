
## deep CCA, not a copy from DeepCCA(CCA cost function). deepCCA, use DNN for projection to maxmium variance explained between projected X and Y (only the first CCA component)
## deep CCA desires to capture all kinds relationships

deepCCA <- function(x,y,devices=mx.cpu(),
                    NeuralNetworkX = list(nlayers=5,
                                          n_hidden=c(50,10,10,10,10),
                                          f_active=c("softrelu","softrelu","softrelu","softrelu","softrelu")),
                    NeuralNetworkY = list(nlayers=1,
                                          n_hidden=c(2),
                                          f_active=c("softrelu")),
                    arg_optimizer=list(name = "adam",
					                   learning.rate = 0.01, 
									   beta1 = 0.9, 
									   beta2 = 0.999,
									   epsilon = 1e-03, 
									   wd = 1e-6, 
									   rescale.grad = 1/1000, 
									   clip_gradient = NULL,
									   lr_scheduler = NULL),
                    arg_initializer = list(scale = 3),
                    f_init = "mx.init.uniform",
                    nit = 5000,
                    track_nit = 50,
					plotit=TRUE,col=NULL,
					printit=TRUE,
					seed=123456
                    ){
					require(mxnet)
                    x=as.matrix(x)
                    y=as.matrix(y)
                    if(nrow(x)!=nrow(y))
                        stop("Error: nrow(x)!=nrow(y).")
                    if(NeuralNetworkX$nlayers!=length(NeuralNetworkX$n_hidden))
                        stop("Error: NeuralNetworkX$nlayers!=length(NeuralNetworkX$n_hidden).")
                    if(NeuralNetworkX$nlayers!=length(NeuralNetworkX$f_active))
                        stop("Error: NeuralNetworkX$nlayers!=length(NeuralNetworkX$f_active).")
                    if(NeuralNetworkY$nlayers!=length(NeuralNetworkY$n_hidden))
                        stop("Error: NeuralNetworkY$nlayers!=length(NeuralNetworkY$n_hidden).")
                    if(NeuralNetworkY$nlayers!=length(NeuralNetworkY$f_active))
                        stop("Error: NeuralNetworkY$nlayers!=length(NeuralNetworkY$f_active).")
                    
                    require(mxnet)
                    X = mx.symbol.Variable('X');
					if(NeuralNetworkY$nlayer==0){
						C <- mx.symbol.FullyConnected(xlayerslist[[NeuralNetworkX$nlayers*2]], num_hidden=1,name=paste0("X","L","F"));
					}else{
						xlayerslist <- list();
						xlayerslist[[1]] <- mx.symbol.FullyConnected(X, num_hidden=NeuralNetworkX$n_hidden[1],name=paste0("X",1,"F"));
						xlayerslist[[2]] <- mx.symbol.Activation(xlayerslist[[1]], act_type=NeuralNetworkX$f_active[1],name=paste0("X",2,"A"));
						if(NeuralNetworkX$nlayer>1){
							for(i in 2:NeuralNetworkX$nlayer){
							   xlayerslist[[2*(i-1)+1]] <- mx.symbol.FullyConnected(xlayerslist[[2*(i-1)]], num_hidden=NeuralNetworkX$n_hidden[i],name=paste0("X",2*(i-1)+1,"F"));
							   xlayerslist[[2*(i-1)+2]] <- mx.symbol.Activation(xlayerslist[[2*(i-1)+1]], act_type=NeuralNetworkX$f_active[i],name=paste0("X",2*(i-1)+2,"A"));
							}
						}
						C <- mx.symbol.FullyConnected(xlayerslist[[NeuralNetworkX$nlayers*2]], num_hidden=1,name=paste0("X","L","F"));
					}
					
					
                    Y = mx.symbol.Variable('Y');
                    ylayerslist <- list();
					if(NeuralNetworkY$nlayer==0){
					    D <- Y
					}else{
					    ylayerslist[[1]] <- mx.symbol.FullyConnected(Y, num_hidden=NeuralNetworkY$n_hidden[1],name=paste0("Y",1,"F"));
						ylayerslist[[2]] <- mx.symbol.Activation(ylayerslist[[1]], act_type=NeuralNetworkY$f_active[1],name=paste0("Y",2,"A"));
						if(NeuralNetworkY$nlayer>1){
							for(i in 2:NeuralNetworkY$nlayer){
							   ylayerslist[[2*(i-1)+1]] <- mx.symbol.FullyConnected(ylayerslist[[2*(i-1)]], num_hidden=NeuralNetworkY$n_hidden[i],name=paste0("Y",2*(i-1)+1,"F"));
							   ylayerslist[[2*(i-1)+2]] <- mx.symbol.Activation(ylayerslist[[2*(i-1)+1]], act_type=NeuralNetworkY$f_active[i],name=paste0("Y",2*(i-1)+2,"A"));
							}
						}
						
						D <- mx.symbol.FullyConnected(ylayerslist[[NeuralNetworkY$nlayers*2]], num_hidden=1,name=paste0("Y","L","F"));
					}
                    
                    
                    loss <- function(C,D){
                              mean1 <- mx.symbol.reshape_like(mx.symbol.dot((C-C+1),mx.symbol.mean(C)),C)
                              mean2 <- mx.symbol.reshape_like(mx.symbol.dot((D-D+1),mx.symbol.mean(D)),D)
                              dev1 <- C-mean1
                              dev2 <- D-mean2
                              dev <- C-D
                              var1 <- mx.symbol.mean(mx.symbol.elemwise_mul(dev1,dev1))
                              var2 <- mx.symbol.mean(mx.symbol.elemwise_mul(dev2,dev2))
							  
							  ##### dif X/var(X)-Y/var(Y)
                              #sumdif/mx.symbol.sqrt(var1*var2+1e-20)
							  var1 <- mx.symbol.reshape_like(mx.symbol.dot((C-C+1),var1),C)
							  var2 <- mx.symbol.reshape_like(mx.symbol.dot((D-D+1),var2),D)
                              #sumdif <- -1*mx.symbol.abs(mx.symbol.sum(mx.symbol.elemwise_mul(dev1,dev2)))
							  #scalse_dif <- C/var1-D/var2
							  scalse_dif <- C/mx.symbol.sqrt(var1)-D/mx.symbol.sqrt(var2)
							  mx.symbol.mean(mx.symbol.square(scalse_dif))
							  
							  
                              #sumdif <- mx.symbol.sum(mx.symbol.elemwise_mul(dev,dev))
							  
							  
                    }

                    E = mx.symbol.MakeLoss(loss(C,D),name="E")
                    
                    batch_size <- nrow(x)
                    
                    a = mx.nd.array(t(as.matrix(x)),devices)
                    b = mx.nd.array(t(as.matrix(y)),devices)

                    data_shape_A <- c(ncol(x),batch_size)
                    data_shape_B <- c(ncol(y),batch_size)
                    
					set.seed(seed)
                    initializer<- do.call(f_init,arg_initializer)
                    arg_param_ini_E<- mx.init.create(initializer = initializer, shape.array = mx.symbol.infer.shape(E, X=data_shape_A,Y=data_shape_B)$arg.shapes, ctx = devices)
                    aux_param_ini_E<- mx.init.create(initializer = initializer, shape.array = mx.symbol.infer.shape(E, X=data_shape_A,Y=data_shape_B)$aux.shapes, ctx = devices)
                    
                    for(i in names(arg_param_ini_E)){
                        tmpd <- as.array(arg_param_ini_E[[i]])
                        tmpd[is.nan(tmpd)] <- runif(sum(is.nan(tmpd)),-3,3);
						tmpd[tmpd>1e3] <- runif(sum(tmpd>1e3),-3,3);
                        tmpd[abs(tmpd)<1e-3] <- runif(sum(abs(tmpd)<1e-3),-3,3);
                        arg_param_ini_E[[i]] <- mx.nd.array(tmpd)
                    }
                    for(i in names(aux_param_ini_E)){
                        tmpd <- as.array(aux_param_ini_E[[i]])
                        tmpd[is.nan(tmpd)] <- runif(sum(is.nan(tmpd)),-3,3);
						tmpd[tmpd>1e3] <- runif(sum(tmpd>1e3),-3,3);
                        tmpd[abs(tmpd)<1e-3] <- runif(sum(abs(tmpd)<1e-3),-3,3);
                        aux_param_ini_E[[i]] <- mx.nd.array(tmpd)
                    }

                    exec_E<- mx.simple.bind(symbol = E, X=data_shape_A,Y=data_shape_B, ctx = devices, grad.req = "write",fixed.param=c("X","Y"))
					exec_C<- mx.simple.bind(symbol = C, X=data_shape_A, ctx = devices, grad.req = "write",fixed.param=c("X"))
					exec_D<- mx.simple.bind(symbol = D, Y=data_shape_B, ctx = devices, grad.req = "write",fixed.param=c("Y"))

					mx.exec.update.arg.arrays(exec_C, arg.arrays = list(X=a), match.name=TRUE)
					mx.exec.update.arg.arrays(exec_D, arg.arrays = list(Y=b), match.name=TRUE)
					
                    mx.exec.update.arg.arrays(exec_E, arg_param_ini_E, match.name=TRUE)
                    mx.exec.update.arg.arrays(exec_E, aux_param_ini_E, match.name=TRUE)

                    input_names_E <- mxnet:::mx.model.check.arguments(E)

                    mx.exec.update.arg.arrays(exec_E, arg.arrays = list(X=a, Y=b), match.name=TRUE)
					
                    mx.exec.forward(exec_E, is.train=T)

                    optimizer_E<-do.call("mx.opt.create",arg_optimizer)
                    
                    updater_E<- mx.opt.get.updater(optimizer = optimizer_E, weights = exec_E$ref.arg.arrays)

                    cor_f <- function(exec_E,exec_C,exec_D){
                        mx.exec.update.arg.arrays(exec_C, exec_E$arg.arrays[names(exec_C$arg.arrays)[-1]], match.name=TRUE)
						mx.exec.update.arg.arrays(exec_D, exec_E$arg.arrays[names(exec_D$arg.arrays)[-1]], match.name=TRUE)
						mx.exec.forward(exec_C)
						mx.exec.forward(exec_D)
						abs(cor(t(as.array(exec_C$outputs[[1]])),t(as.array(exec_D$outputs[[1]]))))
                    }
                    
                    track = numeric()
                    
                    for(i in 1:nit){
                        mx.exec.backward(exec_E)
                        update_args_E_OD <- exec_E$arg.arrays
                        output_OD <- as.array(exec_E$outputs[[1]])
                        update_args_E<- updater_E(weight = exec_E$ref.arg.arrays, grad = exec_E$ref.grad.arrays)
						#update_args_E[grep("Y",names(update_args_E_OD))] <- update_args_E_OD[grep("Y",names(update_args_E_OD))]
                        mx.exec.update.arg.arrays(exec_E, update_args_E, skip.null=TRUE)
                        mx.exec.forward(exec_E, is.train=T)
						mx.exec.backward(exec_E)
                        update_args_E_OD <- exec_E$arg.arrays
                        output_OD <- as.array(exec_E$outputs[[1]])
                        update_args_E<- updater_E(weight = exec_E$ref.arg.arrays, grad = exec_E$ref.grad.arrays)
						#update_args_E[grep("X",names(update_args_E_OD))] <- update_args_E_OD[grep("X",names(update_args_E_OD))]
                        mx.exec.update.arg.arrays(exec_E, update_args_E, skip.null=TRUE)
                        mx.exec.forward(exec_E, is.train=T)
						
                        #track[i,] <- c(as.array(update_args_E[[2]])[1],as.array(update_args_E[[5]])[1])
                        #track[i,] <- c(as.array(update_args_E[[2]])[1],as.array(update_args_E[[2]])[1])
						#output_NEW <- as.array(exec_E$outputs[[1]])
						#if(is.nan(output_NEW)) output_NEW <- output_OD*2+1
                        #if(1/(exp((log(output_NEW)-log(output_OD)))+1)<runif(1)){
                        #    mx.exec.update.arg.arrays(exec_E, update_args_E_OD, skip.null=TRUE)
                        #    mx.exec.forward(exec_E, is.train=T)
                        #}
						
						if(is.nan(sum(as.numeric(as.array(exec_E$outputs[[1]]))))){
							initializer<- do.call(f_init,arg_initializer)
							arg_param_ini_E<- mx.init.create(initializer = initializer, shape.array = mx.symbol.infer.shape(E, X=data_shape_A,Y=data_shape_B)$arg.shapes, ctx = devices)
							for(i in names(arg_param_ini_E)){
									tmpd <- as.array(arg_param_ini_E[[i]])
									tmpd[is.nan(tmpd)] <- runif(sum(is.nan(tmpd)),-3,3);
									tmpd[tmpd>1e3] <- runif(sum(tmpd>1e3),-3,3);
									tmpd[abs(tmpd)<1e-3] <- runif(sum(abs(tmpd)<1e-3),-3,3);
									arg_param_ini_E[[i]] <- mx.nd.array(tmpd)
									mx.exec.update.arg.arrays(exec_E, arg_param_ini_E, match.name=TRUE)
									print("WARN: NAN output, re-initialize weights; Please consider new network structure!")
							
							}
							mx.exec.update.arg.arrays(exec_E, arg_param_ini_E, match.name=TRUE)
							mx.exec.update.arg.arrays(exec_E, arg.arrays = list(X=a, Y=b), match.name=TRUE)
							mx.exec.forward(exec_E, is.train=T)
							optimizer_E<-do.call("mx.opt.create",arg_optimizer)
							updater_E<- mx.opt.get.updater(optimizer = optimizer_E, weights = exec_E$ref.arg.arrays)

						}
						
                        if (printit) print( c(sum(as.array(exec_E$outputs[[1]])),cor_f(exec_E,exec_C,exec_D)))
                        #exec_E$outputs
                        #exec_E$arg.arrays[c(2,5)]
                        if(i%%track_nit==0){
                            track <- rbind(track,c(i,cor_f(exec_E,exec_C,exec_D)))
							if(plotit) plot(t(as.array(exec_C$outputs[[1]])),y,col=col,pch=19,cex=0.5,xlab="deep(X)",ylab="y")
                        }
                    }
					mx.exec.update.arg.arrays(exec_C, exec_E$arg.arrays[names(exec_C$arg.arrays)[-1]], match.name=TRUE)
					mx.exec.update.arg.arrays(exec_D, exec_E$arg.arrays[names(exec_D$arg.arrays)[-1]], match.name=TRUE)
					mx.exec.forward(exec_C);mx.exec.forward(exec_D);
					Xp=as.numeric(as.array(exec_C$outputs[[1]]))
					Yp=as.numeric(as.array(exec_D$outputs[[1]]))
					Xp <- (Xp-mean(Xp))/sd(Xp)
					Yp <- (Yp-mean(Yp))/sd(Yp)
                    #return(list(cor=cor_f(exec_E,NeuralNetworkX,NeuralNetworkY),exec_E,track=track))
					return(list(cor=abs(as.array(exec_E$outputs[[1]])),
					            trained_network=exec_E,
					            project_XY=list(X=Xp,
								                Y=Yp),
								track=track))
}
